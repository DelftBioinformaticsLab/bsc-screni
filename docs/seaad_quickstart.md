# SEA-AD Quickstart

Where to find what files, and what to run for your sub-question. **Use the paired branch as your primary substrate** — the unpaired branch has known integration limitations (see [seaad_integration_history.md](seaad_integration_history.md) if you need the details).

## What's on disk

All under `data/processed/seaad/`. Shared across all sub-questions (Phases 0–2 are one-off):

| File | What it is |
|---|---|
| `seaad_paired_integrated.h5mu` | WNN-integrated paired data, 138k cells × 28 donors. Has the full WNN graph + KNN indices. |
| `seaad_paired_rna_hvg.h5ad` | Full paired cells × 500 HVGs (raw counts). Full SEA-AD obs, joint embedding in `obsm["X_pca"]`. **Most students start here.** |
| `seaad_paired_atac_hvp.h5ad` | Full paired cells × 10k HVPs (raw counts). Same obs/obsm as above. |
| `seaad_unpaired_integrated.h5ad` | 1.5M cells across both modalities, global Harmony embedding. Don't use this. |
| `seaad_unpaired_nn_pairs.csv` | 372k cross-modal pairs (RNA cell ↔ ATAC cell). Don't use this. |
| `seaad_unpaired_donor_summary.csv` | Per-donor diagnostics for the unpaired alignment. Don't use this. |

A **default subsample is pre-baked** on the cluster (seed `42`, 50 cells per all 24 subclasses → 1,200 cells) so you can run Phase 3 out of the box without subsampling first:

| File pattern | What it is |
|---|---|
| `seaad_paired_{rna,atac}_sub42.h5ad` | Default subsample (50/type × 24 types) — Phase 3 input |
| `seaad_paired_sub42_triplets.csv` | (TF, peak, target_gene, spearman_r) — the regulatory triplets |
| `seaad_paired_sub42_gene_labels.csv` | Per-HVG: type = TF or target |
| `seaad_paired_sub42_peak_overlap_matrix.npz` | Cells × correlated peaks (with Gaussian noise) — RF input |
| `seaad_paired_sub42_peak_gene_pairs.csv` | Correlated gene-peak pairs (\|r\| > 0.1) |
| `seaad_paired_sub42_peak_info.csv` | Peak → gene mapping |
| `seaad_paired_sub42_motif_peak_pairs.csv` | MOODS TF-motif hits per peak |

If your sub-question needs a different cell selection (more cells per type, different cell types, a different seed for variance estimation, etc.), make your own subsample with [scripts/subsample_seaad_paired.py](../scripts/subsample_seaad_paired.py) — the seed embeds in the output filename so your subsample doesn't collide with the default `_sub42` or other students'.

## What the scripts do

| Script | What it does | When to run |
|---|---|---|
| [scripts/run_seaad_hvg_selection.py](../scripts/run_seaad_hvg_selection.py) | Reads the 91 GB paired h5mu, writes the `_hvg.h5ad` + `_hvp.h5ad` files | Already done on cluster — re-run only to change HVG/HVP parameters |
| [scripts/subsample_seaad_paired.py](../scripts/subsample_seaad_paired.py) (slurm: [slurm/run_subsample_seaad_paired.sh](../slurm/run_subsample_seaad_paired.sh)) | Subsamples cells from the HVG/HVP files for Phase 3. Args: `--seed`, `--n-per-type`, `--cell-types` | Default (`--seed 42`) already done on cluster; rerun if your sub-question needs a custom selection |
| [scripts/validate_phase3_outputs.py](../scripts/validate_phase3_outputs.py) | Sanity-check Phase 3 outputs (shapes, cross-references, alignment) | After Phase 3, before trusting the output |
| [scripts/inspect_seaad_integration.py](../scripts/inspect_seaad_integration.py) | QC figures (UMAPs) for the integration | When you need to look at integration quality |
| [scripts/check_seaad_atac_counts.py](../scripts/check_seaad_atac_counts.py) | Verifies SEA-AD ATAC `.X` is raw counts | One-off sanity check |
| [scripts/check_moods_in_container.sh](../scripts/check_moods_in_container.sh) | Confirms MOODS is importable in the apptainer image | Before submitting Phase 3 on the cluster |
| [src/screni/seaad_paired_diagnostics.ipynb](../src/screni/seaad_paired_diagnostics.ipynb) | Interactive diagnostics notebook (UMAPs, QC, side-by-side) for the HVG/HVP files | Anytime you want to look at the data |

## The standard sub-question workflow

**If the default subsample (`_sub42`, 50 cells × 24 subclasses) fits your sub-question, the inputs are already on the cluster — just consume them.** Otherwise:

```bash
# 1. (Optional) Make your own subsample with your own seed
sbatch slurm/run_subsample_seaad_paired.sh --seed 123
# Variations (all passed straight through to the python script):
#   --n-per-type 100                                     # bigger sample
#   --cell-types Microglia-PVM Astrocyte                 # only specific subclasses
#   --cell-types Microglia-PVM Astrocyte Oligodendrocyte "L2/3 IT"  # the 4 AD-target types

# 2. Run Phase 3 (globs all sub*.h5ad files automatically;
#    your new sub-files plus any existing ones)
sbatch slurm/run_gene_peak.sh

# 3. Verify outputs are well-formed
pixi run python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub123
```

For wScReNI / cell-specific RF networks, the four files you need are (using `{seed}=42` for the default or your own seed):

- `seaad_paired_rna_sub{seed}.h5ad` — your subsample's RNA expression
- `seaad_paired_sub{seed}_triplets.csv` — TF-peak-target constraint structure
- `seaad_paired_sub{seed}_gene_labels.csv` — which genes are TFs
- `seaad_paired_sub{seed}_peak_overlap_matrix.npz` — peak accessibility (RF input)

## KNN for your cell selection

The HVG/HVP files carry a full-set k=20 WNN KNN in `uns["wnn_neighbor_indices"]`, but those indices reference the full 138k-cell set — **not** your subsample's positions. For your subsample, recompute KNN from `obsm["X_pca"]`:

```python
import anndata as ad
from sklearn.neighbors import NearestNeighbors

rna = ad.read_h5ad("data/processed/seaad/seaad_paired_rna_sub42.h5ad")
knn = NearestNeighbors(n_neighbors=20).fit(rna.obsm["X_pca"])
distances, indices = knn.kneighbors(rna.obsm["X_pca"])
```

`obsm["X_pca"]` is the WNN-input joint embedding (RNA PCA + ATAC LSI, 40 dims), preserved through subsampling.

## Caveats to know

- **Paired vs unpaired**: paired is your default. Unpaired integration has known issues (modality clouds remain disjoint in UMAP, `cell_type_agreement` modest). Don't use the unpaired branch as a primary substrate unless your sub-question is specifically about benchmarking integration.
- **Missing values in obs**: object-dtype obs columns were string-coerced for h5py — `NaN` became `"nan"`, `None` became `"None"`. Detect missingness via `df["col"].isin(["nan", "None", "<NA>"])`, not `.isna()`. Categorical columns (Subclass, Sex, etc.) are unaffected.
- **TF count is bounded by TRANSFAC**, not by HVG selection. Expect ~25 TFs out of any 500-HVG set. The triplet count (~10k) is in line with the paper's retinal benchmark.
- **Cell types not used in Phase 3 correlations**: Phase 3 correlates across all cells in your subsample regardless of type. If you want per-cell-type correlations, subsample one cell type at a time (`--cell-types <one>` with `--seed` per cell type).

## Other docs

- [seaad_integration_history.md](seaad_integration_history.md) — full design history, including the failed unpaired integration iterations and what's still open.
- [processing_pipeline.md](processing_pipeline.md) — general pipeline documentation across all three datasets.
- [project_description.md](project_description.md) — project context and sub-question definitions.
