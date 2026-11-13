# Does SVGRN learn spatially varying regulation, or something simpler?

SVGRN infers a gene regulatory network (GRN) for every cell or spot of a spatial transcriptomics dataset. On simulated data the paper reports that it beats CeSpGRN, locCSN and GRNBoost2 in every setting. We want to know whether to use it on our mapped SEA-AD nuclei, so we ask: does its advantage come from the regulatory and spatial signal in the data, or from something simpler? And how does ScReNI's kScReNI do on the same data?

Paper: Li et al., *Bioinformatics Advances* (2026), doi [10.1093/bioadv/vbag230](https://doi.org/10.1093/bioadv/vbag230). Code: [lyrrrr/SVGRN](https://github.com/lyrrrr/SVGRN) `7ed02a1`. The detailed log, with dates, job IDs and code references, is in [`journal.md`](journal.md).

## Main result so far

**On SVGRN's own simulation, SVGRN is 7% above random, but the same score comes from things that are not regulation or space.** One dataset so far (g110_c2k_n01, 30 cells); replicates are running.

- **Above random, as in the paper.** SVGRN reaches an AUPRC ratio of 1.069 (7% above random; the paper reports 12% for this setting). Low numbers are normal for GRN inference from expression alone, so this alone would be fine.
- **Shuffled RNA does as well.** With every gene shuffled across cells, which destroys all gene–gene structure, SVGRN reaches 11% above random.
- **A rule with no network does as well.** Ranking every TF → gene edge by how highly the target gene is expressed reaches 11% above random. In this simulation, true targets are more highly expressed, and that is what SVGRN's margin reflects.
- **Space adds nothing.** Stage 2 (the spatial, per-cell step) scores 7% above random against 8% for stage 1 alone, and shuffling the positions before stage 2 gives 8%. The paper's own stage-2 ablation (Supplementary Note 2, Table 1) reports only the reconstruction loss, i.e. a better expression fit, not GRN accuracy, so it does not contradict this.
- **kScReNI does as well without space.** ScReNI's RNA-only variant reaches 9% above random, level with SVGRN.
- **The baselines were weakened, but that does not change the picture.** CeSpGRN with the TF prior SVGRN also gets scores the same as without it (4% above random).

## How SVGRN works, in short

- **Input:** expression of every cell, its x, y position, and a list of TFs.
- **Stage 1:** one tissue GRN, learned as a structural equation model inside a conditional VAE. The GRN matrix starts with every TF → gene edge on.
- **Stage 2:** for each target cell, the tissue GRN is fine-tuned with each cell's error weighted by its distance to the target (RBF kernel). Only the GRN matrix changes.
- **Output:** one GRN per cell.

**How we score**, per cell against its true GRN, on TF → gene pairs only (as SVGRN does), then averaged over cells:
- **AUPRC ratio:** AUPRC on |edge weight| divided by the share of true edges, so 1 = random ([`eval.py` L264–266](https://github.com/lyrrrr/SVGRN/blob/7ed02a1/eval.py#L264-L266) in SVGRN's code). We write it as "x% above random".
- **EPR:** precision in the top k edges (k = number of true edges) divided by what random picking gets, so 1 = random ([`eval.py` L226](https://github.com/lyrrrr/SVGRN/blob/7ed02a1/eval.py#L226)).
- SVGRN's own AUPRC ranks by the signed weight, which counts strong repressive edges as non-edges; we report it next to ours.

## The paper's claim

Fig. 3 of the paper: across 6 settings (110 or 160 genes × noise 0.1, 0.5, 1.0; 2,000 cells), SVGRN's AUPRC ratio is 1.06–1.12 (6–12% above random) against 1.01–1.05 for the baselines. The figure has no random baseline and no spread across cells. The Methods say every method got TF–target pairs as prior knowledge; in the released code none does (see [journal](journal.md)).

## Experiments

### 1. SVGRN with controls on its own data · done for g110_c2k_n01

| method | AUPRC ratio | % above random |
|---|---|---|
| target gene mean expression (no network) | 1.110 ± 0.111 | 11% |
| SVGRN on shuffled RNA | 1.107 ± 0.100 | 11% |
| kScReNI (expression neighbours + GENIE3, no space) | 1.088 ± 0.088 | 9% |
| SVGRN stage 1 only | 1.084 ± 0.097 | 8% |
| SVGRN, positions shuffled | 1.081 ± 0.095 | 8% |
| **SVGRN as published** | **1.069 ± 0.090** | **7%** |
| CeSpGRN with TF prior, 8 settings averaged | 1.042 ± 0.073 | 4% |
| CeSpGRN as SVGRN ran it | 1.040 ± 0.063 | 4% |
| Pearson in SVGRN's neighbourhood | 1.039 ± 0.079 | 4% |
| random | 1.010 ± 0.058 | 1% |
| Pearson over all cells | 0.983 ± 0.049 | −2% |

Mean ± SD over 30 target cells. Standard error of a mean ≈ 0.016, so differences under about 0.03 are noise.

### 2. Replicates: 6 settings regenerated with scMultiSim · running

SVGRN's own scMultiSim scripts and settings, seed 0. They do not reproduce SVGRN's files bit for bit (likely a different scMultiSim version), so they are new datasets with the paper's settings. Same evaluation as experiment 1.

### 3. kScReNI on the same data · done for g110_c2k_n01, running on the replicates

ScReNI's RNA-only variant: 20 expression neighbours per cell, GENIE3 on those 21 cells, run unmodified. On g110_c2k_n01 it reaches 9% above random (1.088 ± 0.088), level with SVGRN within noise, without positions or a GPU (2,000 cells in about a minute on 16 CPUs).

## Plan

1. Experiment 1 on SVGRN's data · done for one of six datasets.
2. Experiments 1 and 3 on the 6 regenerated datasets · running.
3. More seeds per setting, if the 6 settings disagree.
4. Decide for SEA-AD: if SVGRN keeps no advantage over its controls, use a simpler method on depth-mapped nuclei.

## Setup

- Branch `feature/svgrn-eval` of Timo's `bsc-screni`, draft [PR #15](https://github.com/DelftBioinformaticsLab/bsc-screni/pull/15). Code in `scripts/svgrn_eval/`, DAIC jobs in `slurm/svgrn_*.sh`.
- SVGRN, CeSpGRN and ScReNI are pinned submodules in `external/`, run unmodified. Small shims let the 2024-era code run on torch 2 (`Variable(name=)`, `torch.load` weights_only, `torch.cholesky`, `torch.eig`).
- Environments: `svgrn` (torch, CUDA 12), `scmultisim` (R, scMultiSim 1.6.0), `kscreni` (R, Seurat, GENIE3), in `pixi.toml`.
- Data: SVGRN's simulations from the README's Google Drive link, and our regenerated ones, on DAIC under `/tudelft.net/staff-umbrella/ScReNI/dflam/`.

## Limits

- One dataset so far; replicates are running.
- 30 target cells per dataset, not all 2,000.
- Simulated data only. These simulations may hold little recoverable network signal for any method (Pearson over all cells is below random), so they cannot show that a method works on real tissue, only whether the paper's claim holds on its own benchmark.
- Our CeSpGRN kernel stands in for the unpublished `calc_kernel_distance`.
- The released stage-2 script uses bandwidth W = 6; the supplement says 4 for benchmarks. Our main runs use 6; a W = 4 run on g110_c2k_n01 checks the difference (job 13034485).
