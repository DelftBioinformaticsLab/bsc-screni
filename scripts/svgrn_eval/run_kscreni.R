# Run ScReNI's kScReNI (external/ScReNI/R/Infer_kScReNI_scNetworks.R, unmodified)
# on one simulated dataset and write the networks of the target cells.
#
# kScReNI is RNA-only: for every cell it takes the cell's knn = 20 nearest
# neighbours on Seurat's shared-nearest-neighbour graph (expression PCA, not
# space) and runs GENIE3 (100 trees) on those 21 cells. As in the function,
# the input is the raw count matrix (genes x cells). Cells are independent,
# so all cells are run as published and only the targets are written.
#
# Usage: Rscript scripts/svgrn_eval/run_kscreni.R <dataset dir> <targets.txt> <out dir> [nthread]
# Writes <out>/kscreni/cells/<cell>.csv (regulator rows x target columns),
# then run collect_csv_preds.py to make preds_kscreni.npz.

args <- commandArgs(trailingOnly = TRUE)
ds <- args[1]; targets <- readLines(args[2]); out <- args[3]
nthread <- if (length(args) >= 4) as.integer(args[4]) else 8
script_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "..", "..", "external", "ScReNI", "R", "Infer_kScReNI_scNetworks.R"))

raw <- read.csv(file.path(ds, "raw_count.csv"), row.names = 1, check.names = FALSE)
# raw_count.csv's gene names are broken in SVGRN's published data (see
# run_cespgrn.py); columns are genes 1..n in order, so name them by position.
genes <- as.character(seq_len(ncol(raw)))
expr <- t(as.matrix(raw))
rownames(expr) <- paste0("g", genes)   # Seurat dislikes purely numeric feature names
colnames(expr) <- rownames(raw)

t0 <- Sys.time()
nets <- Infer_kScReNI_scNetworks(expr, nfeatures = 4000, knn = 20, nthread = nthread, nTrees = 100)
cat(sprintf("kScReNI: %d cells in %.1f min\n", length(nets), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

dir.create(file.path(out, "kscreni", "cells"), recursive = TRUE, showWarnings = FALSE)
for (cell in targets) {
  m <- nets[[cell]]
  m <- m[paste0("g", genes), paste0("g", genes)]
  dimnames(m) <- list(genes, genes)
  write.csv(m, file.path(out, "kscreni", "cells", paste0(cell, ".csv")))
}
cat(sprintf("wrote %d target networks to %s\n", length(targets), file.path(out, "kscreni", "cells")))
