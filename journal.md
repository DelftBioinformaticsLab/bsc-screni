# Journal

**Question:** SVGRN claims better cell-specific GRNs from spatial transcriptomics than CeSpGRN, locCSN and GRNBoost2. Does that hold on its own simulated data once the baselines are fair and simple controls are added? Does the spatial part add anything? And how does ScReNI's kScReNI compare on the same data?

This file is the working log: each finding with how we got it (dates, job IDs, code lines). Newest entries go at the bottom. The short version for readers is [`research.md`](research.md).

Paper: Li, Chen, Lu, Tsai & Wang, *Spatially varying gene regulation network inference from spatial transcriptomics*, Bioinformatics Advances (2026), doi [10.1093/bioadv/vbag230](https://doi.org/10.1093/bioadv/vbag230) (PMC13615653; preprint bioRxiv 10.1101/2025.10.17.683188). Code: [lyrrrr/SVGRN](https://github.com/lyrrrr/SVGRN) at `7ed02a1`, submodule `external/SVGRN`. Baselines: [PeterZZQ/CeSpGRN](https://github.com/PeterZZQ/CeSpGRN) `2fb222f` (v1.0.0), [Xuxl2020/ScReNI](https://github.com/Xuxl2020/ScReNI) `695772f`.

All metrics are ratios to a random predictor (1 = random), scored per target cell against that cell's true GRN on TF → gene pairs, then averaged over cells. We write results as "x% above random".

## Journal

### 2026-11-13 · How SVGRN works in the code

**Finding.** Stage 1 learns one tissue GRN (DeepSEM-style SEM inside a CVAE); the GRN matrix starts with every TF → gene edge set to 1. Stage 2 copies it for each target cell and fine-tunes only that matrix, weighting each cell's reconstruction error with an RBF kernel on its distance to the target (σ = W = 6 in raw coordinates). Every cell's position input is replaced by the target's position, so space enters stage 2 only through those weights.

**How.** `src/SVGRN_allcell.py:22-26` (`initalize_A_withTF`), `src/SVGRN_singlecell.py:60-95` (RBF weights, positions set to the target), `:160-168` (only `adj_A` is optimised).

### 2026-11-13 · The paper says every method got a TF–target prior; the code gives none

**Finding.** Methods: "TF–target pairs were provided as shared prior knowledge across all methods to ensure fair comparison." In the released code no method gets TF–target pairs: SVGRN gets a TF list, GRNBoost2 a hard-coded TF list, CeSpGRN and locCSN nothing. What all methods share is the scoring, restricted to TF → gene pairs. In these simulations a TF–target prior would hardly differ from a TF list: `allcells_GRN.csv` has 600 nonzero edges of ~650 possible TF → gene pairs (g110 datasets).

**How.** `main_stage1.py` (`--tf_list`); `benchmark/cespgrn/run_cespgrn.py` (`G_admm_minibatch(X=..., K=K, pre_cov=empir_cov, batchsize=120)`, no `TF=`); `benchmark/grnboost2/run_grnboost2.py` (`tf_list = ["5", "8", ...]`); `benchmark/locCSN/run_locCSN.py` (gets `ClusterID`, the true GRN clusters); `eval.py:11-27` (`get_evaluation_matrix`).

### 2026-11-13 · The CeSpGRN baseline calls a function that does not exist in CeSpGRN

**Finding.** `run_cespgrn.py` calls `kernel.calc_kernel_distance`, which is in no CeSpGRN release or commit. The script cannot be exactly what produced Fig. 3. It also runs one fixed setting (bandwidth 0.1, λ 0.1, 30 neighbours) instead of CeSpGRN's average over settings.

**How.** `git log -S calc_kernel_distance --all` on CeSpGRN: no hits.

### 2026-11-13 · Their AUPRC uses signed weights; cells without output are skipped

**Finding.** `get_AUPRC_AUROC` ranks edges by the signed weight, so strong repressive edges count as non-edges; early precision uses |weight|. `eval_scGRN_SVGRN` silently skips target cells whose result file is missing. We score on |weight| and report the signed version next to it.

**How.** `eval.py:251-252` (`y_score = A[Evaluate_Mask]`), `eval.py:201` (abs), `eval.py:160-161` (`continue`).

### 2026-11-13 · Both metrics are ratios to random

**Finding.** The paper uses BEELINE's EPR and AUPRC ratio. In the code, `auprc_ratio = auprc / prevalence` (random ranking has AUPRC = prevalence) and EPR = precision in the top k / (k / number of candidate pairs). So 1 = random. Fig. 3: SVGRN AUPRC ratio 1.06–1.12 (6–12% above random), CeSpGRN 1.03–1.04, locCSN 1.03–1.05, GRNBoost2 1.01–1.04; no random baseline and no spread across cells in the figure.

**How.** `eval.py:264-266` and `eval.py:226`; paper Section 3, Fig. 3.

### 2026-11-13 · Published data has broken gene names in raw_count.csv

**Finding.** `raw_count.csv` names genes 1..100, then 102–105, then `gene105`..`gene110` (101 missing, 105 twice). The columns are in order: per-gene correlation of log-normalised raw counts with SVGRN's input expression is median 1.000, min 0.999. SVGRN's own CeSpGRN script reads it by position, so we do too.

**How.** `scripts/svgrn_eval/run_cespgrn.py` (check printed in `log_cespgrn.txt`).

### 2026-11-13 · First evaluation on g110_c2k_n01: the margin over random comes from per-gene properties

**Finding.** On SVGRN's own dataset (110 genes, noise 0.1, 2,000 cells; 30 target cells stratified by GRN cluster), AUPRC ratio on |weights|, mean ± SD over cells:

| method | AUPRC ratio | % above random |
|---|---|---|
| target gene mean expression (no gene–gene information) | 1.110 ± 0.111 | 11% |
| SVGRN on shuffled RNA (stage 2) | 1.107 ± 0.100 | 11% |
| SVGRN stage 1 only (tissue GRN) | 1.084 ± 0.097 | 8% |
| SVGRN, positions shuffled before stage 2 | 1.081 ± 0.095 | 8% |
| SVGRN as published (stage 2) | 1.069 ± 0.090 | 7% |
| CeSpGRN, TF prior, 8-setting average | 1.042 ± 0.073 | 4% |
| CeSpGRN as SVGRN ran it | 1.040 ± 0.063 | 4% |
| Pearson in SVGRN's RBF neighbourhood | 1.039 ± 0.079 | 4% |
| random | 1.010 ± 0.058 | 1% |
| Pearson over all cells | 0.983 ± 0.049 | −2% |

SVGRN reaches 7% above random (paper: 12% for this setting). The same model on shuffled RNA (every gene permuted across cells) reaches 11%, and ranking TF → gene edges by the target gene's mean expression alone reaches 11%. So SVGRN's margin over random is explained by a per-gene property of the simulation, not by regulatory signal. Stage 2 does not improve on stage 1, and shuffling positions before stage 2 does not hurt it: the spatial part adds nothing here. Giving CeSpGRN the TF prior changes nothing. SE of a mean ≈ 0.016, so differences under ~0.03 are noise.

**How.** `slurm/svgrn_01_run.sh` with `DS=g110_c2k_n01 N_TARGETS=30`: jobs 13034313 (published, baselines, CeSpGRN), 13034314 (shuffled RNA), 13034315 (shuffled positions); `scripts/svgrn_eval/score.py`. Stage 1 took 32 s, stage 2 a median 24 s per cell on one GPU. Output: DAIC `bsc-screni-svgrn/output/svgrn_eval/g110_c2k_n01/scores_summary.csv`.

### 2026-11-13 · Regenerated the simulations with SVGRN's scMultiSim scripts

**Finding.** `simulate.R` ports `scMultiSim_DataPrepare/g110_c2k_0.1.R` and `g160_c2k_0.1.R` (same GRN, ligand pairs, options; genes, noise and seed as parameters), `prepare_sim.py` ports `sim_prepare.py`. All 6 settings ran in 2–3 min each. Seed 0 does **not** reproduce SVGRN's downloaded files: counts, locations and cell GRNs differ (count correlation ≈ 0), with the same structure (2,000 cells, 6 TFs `2 6 10 19 80 91` for 110 genes, layers layout). The paper does not state its scMultiSim version; ours is 1.6.0. We treat the regenerated data as independent replicates with the paper's settings.

**How.** `slurm/svgrn_00_simulate.sh` (`SEED=0`, array job 13034398); comparison in the session log (raw counts, `cell_loc_GRNcluster4.csv`, `cell_specific_GRN/cell7.csv`).

### 2026-11-13 · kScReNI added

**Finding.** ScReNI's kScReNI is RNA-only: for each cell, its 20 nearest neighbours on Seurat's SNN graph (expression PCA, not space), then GENIE3 (100 trees) on those 21 cells, input the raw counts. Run unmodified on all cells (2,000 cells in 1.1 min on 16 CPUs); the target cells are scored like the other methods. On g110_c2k_n01: AUPRC ratio 1.088 ± 0.088 (9% above random), AUROC 0.511, EPR 1.087. That is level with SVGRN (7%) within noise, with no spatial information and no GPU, and still below the no-network rule (11%).

**How.** `external/ScReNI/R/Infer_kScReNI_scNetworks.R`; `scripts/svgrn_eval/run_kscreni.R`, `slurm/svgrn_02_kscreni.sh` (job 13034419 on g110_c2k_n01; jobs 13034429–13034461 on the regenerated datasets).

### 2026-11-13 · The paper's stage-2 ablation measures expression fit, not GRN accuracy

**Finding.** Supplementary Note 2 and Table 1 are the paper's only evidence that stage 2 helps. They report the reconstruction loss, not EPR or AUPRC: g110 noise 0.1, 1.4846 with neither stage-2 component → 1.2652 with both; noise 0.5, 1.8651 → 1.6961. A better expression fit around the target does not mean a better GRN, so it does not contradict our result that stage 2 does not improve GRN accuracy (7% vs 8% above random for stage 1). The losses are also not on the same footing: the kernel-weighted loss (`weighted_reconstruction_loss`) reweights cells toward the target's neighbours, which stage 2 is fitted to.

**How.** Supplement `SVGRN_Supplementary_Material_Clean.pdf` p. 2–3 (Note 2, Table 1); `src/Con_Model_newED.py:31-42`. Raised by PHOENIX from the supplement; checked against the PDF.

### 2026-11-13 · The supplement's bandwidth differs from the released script

**Finding.** Supplementary Note 4: "The bandwidth parameter was set to 4 for benchmarking experiments". The released stage-2 script uses `--W 6`, which is what our runs used. We add a run with W = 4 (same stage-1 model, same targets) to check that this does not change the result.

**How.** Supplement p. 3 (Note 4); `train_main_stage2_allcell.sh` (`--W 6`); `run_svgrn.py --W 4 --name published_w4`.
