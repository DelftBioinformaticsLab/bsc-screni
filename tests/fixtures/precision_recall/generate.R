## Regenerate the parity fixture. Run from the repository root:
##   Rscript tests/fixtures/precision_recall/generate.R
##
## Requires CRAN packages: plyr, reshape2.
##
## Sources the original ScReNI R functions from:
##   https://github.com/Xuxl2020/ScReNI/master/R/

suppressPackageStartupMessages({
  library(plyr)
  library(reshape2)
})

FIXTURE_DIR <- file.path("tests", "fixtures", "precision_recall")
R_SRC <- file.path(FIXTURE_DIR, ".r_src")
dir.create(R_SRC, showWarnings=FALSE)

base <- "https://raw.githubusercontent.com/Xuxl2020/ScReNI/master/R"
for (f in c("Calculate_scNetwork_precision_recall.R",
            "Calculate_scNetwork_precision_recall_top.R",
            "Precision_recall_affiliated_functions.R")) {
  dest <- file.path(R_SRC, f)
  if (!file.exists(dest)) {
    download.file(file.path(base, f), dest, quiet=TRUE)
  }
  source(dest)
}

set.seed(42)
genes <- c("a","b","c","d","e")
n_cells <- 4

make_net <- function(density) {
  M <- matrix(runif(length(genes)^2), nrow=length(genes), dimnames=list(genes, genes))
  diag(M) <- 0
  M[M < (1 - density)] <- 0
  M
}

scNetworks <- list(
  CSN     = lapply(seq_len(n_cells), function(i) make_net(0.5)),
  wScReNI = lapply(seq_len(n_cells), function(i) make_net(0.9)),
  kScReNI = lapply(seq_len(n_cells), function(i) make_net(0.9))
)
for (n in names(scNetworks)) {
  names(scNetworks[[n]]) <- paste0("cell", seq_len(n_cells))
}

TF_target_pair <- c("a_b", "a_c", "b_d", "c_e", "d_a")

## ---- save inputs (long-form so Python reads identical floats) ----
flat <- do.call(rbind, lapply(names(scNetworks), function(net) {
  do.call(rbind, lapply(names(scNetworks[[net]]), function(cell) {
    df <- reshape2::melt(scNetworks[[net]][[cell]],
                         varnames=c("from","to"), value.name="weight")
    df$net  <- net
    df$cell <- cell
    df
  }))
}))
write.csv(flat, file.path(FIXTURE_DIR, "scNetworks_long.csv"), row.names=FALSE)
writeLines(TF_target_pair, file.path(FIXTURE_DIR, "TF_target_pair.txt"))

## ---- run R reference and snapshot every output ----
top_number <- c(5, 10)
res <- Calculate_scNetwork_precision_recall(
  scNetworks, TF_target_pair, top_number=top_number,
  gene_id_gene_name_pair=NULL, gene_name_type=NULL
)
for (k in names(res)) {
  df <- res[[k]]
  df$cell <- rownames(df)
  write.csv(df, file.path(FIXTURE_DIR, sprintf("r_pr_%s.csv", k)), row.names=FALSE)
}

## NB: pass res with explicit key lookup so the R bug (positional [[i]]) does
## not corrupt the comparison. The Python implementation looks up by key.
res_for_top <- res[as.character(top_number)]
top_res <- Calculate_scNetwork_precision_recall_top(res_for_top, top_number)
write.csv(top_res[[1]], file.path(FIXTURE_DIR, "r_precision_top.csv"), row.names=FALSE)
write.csv(top_res[[2]], file.path(FIXTURE_DIR, "r_recall_top.csv"),    row.names=FALSE)

## summarySE direct fixture
df_se <- data.frame(
  g = factor(c("a","a","a","b","b","b","b")),
  x = c(1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0)
)
write.csv(df_se, file.path(FIXTURE_DIR, "se_input.csv"), row.names=FALSE)
se <- summarySE(df_se, measurevar="x", groupvars="g")
write.csv(se, file.path(FIXTURE_DIR, "r_se.csv"), row.names=FALSE)

cat("Fixture regenerated under", FIXTURE_DIR, "\n")
