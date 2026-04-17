# Run the paper's exact Phase 3 pipeline and report intermediate counts.
# Requires: Seurat, Signac, Matrix, motifmatchr, BSgenome.Mmusculus.UCSC.mm10
#
# Usage (from WSL):
#   .libPaths(c("~/R/libs", .libPaths()))
#   source("scripts/run_paper_phase3.R")

.libPaths(c("~/R/libs", .libPaths()))

library(Matrix)
library(SeuratObject)
library(GenomicRanges)

# Create a minimal Signac stub package in a temp dir so R's class
# resolver finds it.  readRDS triggers requireNamespace("Signac")
# when deserializing ChromatinAssay objects.
stub_dir <- file.path(tempdir(), "Signac")
dir.create(file.path(stub_dir, "R"), recursive = TRUE, showWarnings = FALSE)
writeLines(c("Package: Signac", "Version: 0.0.1", "Title: stub"),
           file.path(stub_dir, "DESCRIPTION"))
writeLines('exportPattern(".")', file.path(stub_dir, "NAMESPACE"))
writeLines(c(
    'setClass("ChromatinAssay", contains = "list")',
    'setClass("Fragments", contains = "list")'
), file.path(stub_dir, "R", "stub.R"))
install.packages(stub_dir, repos = NULL, type = "source",
                 lib = "~/R/libs", quiet = TRUE)

# Source the ScReNI functions
screni_dir <- "/mnt/c/Users/timov/Repositories/bsc-screni/ext/ScReNI/R"
source(file.path(screni_dir, "wScReNI_affiliated_functions.R"))
source(file.path(screni_dir, "Infer_gene_peak_relationships.R"))

paper_dir <- "/mnt/c/Users/timov/Repositories/bsc-screni/data/paper/datasets"
ref_dir <- "/mnt/c/Users/timov/Repositories/bsc-screni/data/paper/reference"

# ── Load the data ──────────────────────────────────────────────────────────
cat("Loading Seurat objects...\n")
rna_obj <- readRDS(file.path(paper_dir, "mmRetina_RPCMG_scRNAseq.rds"))
cat("RNA loaded\n")
atac_obj <- readRDS(file.path(paper_dir, "mmRetina_RPCMG_scATACseq.rds"))
cat("ATAC loaded\n")

# ── Load annotation (the 400 subsampled cells) ────────────────────────────
annotation <- read.csv(file.path(paper_dir, "mmRetina_RPCMG_Cell100_annotation.csv"),
                       row.names = 1)
cat("Annotation:", nrow(annotation), "cells\n")
cat("Cell types:", table(annotation$undup.cell.types), "\n")

# ── Extract the subsampled RNA and ATAC matrices ──────────────────────────
# The annotation has RNA cell names and ATAC cell names
rna_cells <- annotation$undup.rna.points.id
atac_cells <- annotation$undup.atac.points.id

# Get counts layers
find_counts <- function(assay) {
    if (.hasSlot(assay, "layers")) {
        lyrs <- slot(assay, "layers")
        if ("counts" %in% names(lyrs)) return(lyrs[["counts"]])
    }
    if (.hasSlot(assay, "counts")) return(slot(assay, "counts"))
    if (is.list(assay)) {
        for (nm in names(assay)) {
            if (inherits(assay[[nm]], "dgCMatrix")) return(assay[[nm]])
        }
    }
    stop("Could not find counts")
}

rna_counts <- find_counts(rna_obj@assays[["RNA"]])
atac_counts <- find_counts(atac_obj@assays[["ATAC"]])

cat("RNA counts:", nrow(rna_counts), "x", ncol(rna_counts), "\n")
cat("ATAC counts:", nrow(atac_counts), "x", ncol(atac_counts), "\n")

# Subset to the 400 cells
sub_rna <- rna_counts[, rna_cells]
sub_atac <- atac_counts[, atac_cells]

# Align column names (paper does: colnames(scatac) = colnames(scrna))
colnames(sub_atac) <- colnames(sub_rna)

# Filter ATAC to chr peaks
sub_atac <- sub_atac[grep("chr", rownames(sub_atac)), ]

# Standardize peak names: chr1-start-end -> chr1:start-end
# (GRanges requires colon separator)
old_peaks <- rownames(sub_atac)
new_peaks <- sub("^(chr[^-]+)-(\\d+)-(\\d+)$", "\\1:\\2-\\3", old_peaks)
rownames(sub_atac) <- new_peaks

cat("\nSubsampled RNA:", nrow(sub_rna), "x", ncol(sub_rna), "\n")
cat("Subsampled ATAC:", nrow(sub_atac), "x", ncol(sub_atac), "\n")
cat("ATAC max:", max(sub_atac@x), "\n")
cat("ATAC peak sample:", head(rownames(sub_atac), 3), "\n")

# ── Feature selection (select_features) ───────────────────────────────────
source(file.path(screni_dir, "select_features.R"))

cat("\n=== Feature Selection ===\n")
sub_rna_top <- select_features(sub_rna, nfeatures = 500, datatype = "RNA")
sub_atac_top <- select_features(sub_atac, nfeatures = 10000, datatype = "ATAC")

cat("RNA after HVG:", dim(sub_rna_top), "\n")
cat("ATAC after HVP:", dim(sub_atac_top), "\n")
cat("RNA genes (first 10):", head(rownames(sub_rna_top), 10), "\n")

# ── Phase 3a: Peak-gene overlap ──────────────────────────────────────────
cat("\n=== Phase 3a: Peak-gene overlap ===\n")
gtf_data <- as.data.frame(rtracklayer::import(file.path(ref_dir, "mouse.genes.gtf")))

peak_gene_overlap_GR <- peak_gene_overlap_GR1(
    gtf_data = gtf_data,
    scrna = sub_rna_top,
    gene_name_type = "symbol",
    scatac = sub_atac_top,
    upstream_len = 250000,
    downstream_len = 250000
)

cat("Overlap pairs:", length(peak_gene_overlap_GR), "\n")
cat("Unique genes:", length(unique(peak_gene_overlap_GR$overlap_gene)), "\n")
cat("Unique peaks:", length(unique(peak_gene_overlap_GR$peak_id)), "\n")

# ── Phase 3b: Correlation filtering ──────────────────────────────────────
cat("\n=== Phase 3b: Correlation filtering ===\n")
overlap_gene_peak1 <- gene_peak_corr1(
    exprMatrix = as.matrix(sub_rna_top),
    scatac = sub_atac_top,
    peak_gene_overlap_GR = peak_gene_overlap_GR,
    threshold = 0.1,
    nthread = 4
)

cat("Correlated pairs:", nrow(overlap_gene_peak1), "\n")
cat("Unique genes:", length(unique(overlap_gene_peak1[, 1])), "\n")
cat("Unique peaks:", length(unique(overlap_gene_peak1[, 2])), "\n")

# ── Report (skip motif matching - needs BSgenome + motifmatchr) ──────────
cat("\n=== Summary ===\n")
cat("Phase 3a overlap pairs:", length(peak_gene_overlap_GR), "\n")
cat("Phase 3b correlated pairs:", nrow(overlap_gene_peak1), "\n")

# Save for comparison
write.csv(
    data.frame(
        gene = overlap_gene_peak1[, 1],
        peak = overlap_gene_peak1[, 2]
    ),
    file.path(paper_dir, "r_correlated_pairs.csv"),
    row.names = FALSE
)

# Save HVGs and HVPs for comparison
writeLines(rownames(sub_rna_top), file.path(paper_dir, "r_hvg_500.txt"))
writeLines(rownames(sub_atac_top), file.path(paper_dir, "r_hvp_10000.txt"))

# Also save the overlap pairs (Phase 3a) for comparison
overlap_df <- data.frame(
    gene = peak_gene_overlap_GR$overlap_gene,
    peak = peak_gene_overlap_GR$peak_id
)
write.csv(overlap_df, file.path(paper_dir, "r_overlap_pairs.csv"), row.names = FALSE)

cat("\nSaved r_correlated_pairs.csv, r_hvg_500.txt, r_hvp_10000.txt, r_overlap_pairs.csv\n")
cat("Done!\n")
