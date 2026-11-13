# Generate one simulated spatial dataset exactly as SVGRN did
# (external/SVGRN/scMultiSim_DataPrepare/g110_c2k_0.1.R and g160_c2k_0.1.R).
# The two scripts differ only in the GRN (scMultiSim's GRN_params_100 for 110
# genes, GRN_files/g150_TF9.csv for 160), the ligand-receptor gene ids and
# num.genes; noise (intrinsic.noise) and the seed are parameters here.
#
# Usage: Rscript scripts/svgrn_eval/simulate.R <genes 110|160> <noise> <seed> <out_dir>
# Writes <out_dir>/{cell_loc.csv, raw_count.csv, cell_specific_GRN/cell<i>.csv}
# (same files and layout as the SVGRN script), then run prepare_sim.py.

suppressPackageStartupMessages(library("scMultiSim"))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: simulate.R <genes 110|160> <noise> <seed> <out_dir>")
n_genes <- as.integer(args[1]); noise <- as.numeric(args[2])
seed <- as.integer(args[3]); out <- args[4]
n_cells <- 2000

script_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
svgrn_prep <- file.path(script_dir, "..", "..", "external", "SVGRN", "scMultiSim_DataPrepare")

if (n_genes == 110) {
  data(GRN_params_100)
  GRN_params <- GRN_params_100
  lig_params <- data.frame(target = c(102, 103), regulator = c(104, 105), effect = c(5.2, 5.9))
} else if (n_genes == 160) {
  GRN_params <- read.csv(file.path(svgrn_prep, "GRN_files", "g150_TF9.csv"))
  lig_params <- data.frame(target = c(152, 153), regulator = c(154, 155), effect = c(5.2, 5.9))
} else stop("genes must be 110 or 160 (the two SVGRN settings)")

spatial_options <- function(...) {
  cci_opt <- list(params = lig_params, max.neighbors = 4, cell.type.interaction = "random")
  list(
    rand.seed = seed,
    GRN = GRN_params,
    num.genes = n_genes,
    num.cells = n_cells,
    num.cifs = 50,
    tree = Phyla1(),
    diff.cif.fraction = 0.8,
    do.velocity = FALSE,
    speed.up = TRUE,
    intrinsic.noise = noise,
    dynamic.GRN = list(num.steps = 1, cell.per.step = 1, num.changing.edges = 5,
                       weight.mean = 0, weight.sd = 4),
    cci = c(cci_opt, list(...))
  )
}

t0 <- Sys.time()
results <- sim_true_counts(spatial_options(layout = "layers"))
dir.create(file.path(out, "cell_specific_GRN"), recursive = TRUE, showWarnings = FALSE)
write.csv(results[["cci_locs"]], file.path(out, "cell_loc.csv"), row.names = TRUE)
write.csv(t(results[["counts"]]), file.path(out, "raw_count.csv"), row.names = TRUE)
for (x in 1:n_cells) {
  write.csv(results[["cell_specific_grn"]][[x]],
            file.path(out, "cell_specific_GRN", paste0("cell", x, ".csv")), row.names = TRUE)
}
cat(sprintf("simulated %d genes, noise %s, seed %d in %.1f min -> %s\n", n_genes, noise, seed,
            as.numeric(difftime(Sys.time(), t0, units = "mins")), out))
cat("scMultiSim version:", as.character(packageVersion("scMultiSim")), "\n")
