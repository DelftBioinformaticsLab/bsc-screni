# RP: Network Representations of Single Cells to Understand Alzheimer's Disease

Python reimplementation of the [ScReNI](https://github.com/Xuxl2020/ScReNI)
single-cell regulatory network inference pipeline (Xu et al. 2025, GPB),
extended to the SEA-AD Alzheimer's dataset.

## Setting Up Pixi on the HPC Cluster

### 1. Log in to the cluster

```bash
ssh [netID]@login.daic.tudelft.nl
login using netID password
```

### 2. Using pixi
Pixi is a fast, modern, and reproducible package management tool for developers of all backgrounds. It replaces conda/mamba as an environment manager by having the entire environment and dependencies all saved within the project directory, making it more modular and reproducible.

Pixi is **not** available on the cluster by default, but we can use a premade container to run it in (see below)

Locally, you should follow the tutorial from the Pixi website: https://pixi.prefix.dev/latest/installation/

### 3. Install the project environment

Navigate to your copy of the repository and install all dependencies declared in `pixi.toml`:

```bash
cd /tudelft.net/staff-umbrella/ScReNI/YOUR_NETID/bsc-screni
pixi install
```

This resolves and installs the exact locked environment in one step — no manual package management needed.

### 4. Run pipeline commands

All pipeline steps are exposed as named tasks:

```bash
pixi run load-paper
pixi run feature-select
pixi run gene-peak
# etc.
```
These scripts have been run already and output has been stored in /tudelft.net/staff-umbrella/ScReNI/bsc-screni. Either make calls to those files (but don't make changes to them as they might be shared between users!) or run the pipeline your own, which is a good way to get familiar with the preprocessing.

### 5. Running SLURM jobs with a container

For cluster jobs, use an Apptainer container instead of activating pixi directly. This avoids environment activation issues inside SLURM and gives fully reproducible runs.

**Premade container:** A ready-to-use `.sif` image is available at:

```
/tudelft.net/staff-umbrella/ScReNI/bsc-screni/*.sif
```

**Building your own container:** If you need to customise the image (e.g. add a package), read docs/using_containers.md

**Example SLURM script using the container:**

```bash
#! /bin/sh
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --cpus-per-task=34
#SBATCH --mem=24000
#SBATCH --time=12:57:59
#SBATCH --job-name=screni
#SBATCH --mail-user=netid@student.tudelft.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

SIF=/tudelft.net/staff-umbrella/ScReNI/bsc-screni/screni.sif

apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" python main.py
```

DAIC documentation can be found at https://daic.tudelft.nl/

### 6. Accessing the shared project folder

The shared project data is located at:

```
/tudelft.net/staff-umbrella/ScReNI
```

You can reference this path in your scripts to read input data or write results.


## Project structure

```
src/screni/data/
    loading.py               Phase 0 PBMC: load 10X Multiome, CellTypist annotation, QC
    loading_paper.py         Phase 0 retinal: load paper's Seurat exports (MTX + metadata)
    loading_seaad.py         Phase 0 SEA-AD: load from AWS, split paired/unpaired
    integration.py           Phase 1 PBMC: WNN integration (also used by SEA-AD paired)
    integration_seaad.py     Phase 1 SEA-AD: WNN (paired) + per-donor Harmony (unpaired)
    feature_selection.py     Phase 2: cell subsampling + HVG/HVP selection
    gene_peak_relations.py   Phase 3: peak-gene overlap, correlation, motif matching, triplets
    utils.py                 Shared: peak parsing, GTF loading, gene activity computation

scripts/
    run_paper_phase3.R       Run paper's R code for Phase 2-3 (validation / R-reference features)

data/paper/
    datasets/                Paper's Seurat exports + R-exported feature lists
    reference/               TRANSFAC motifs, PWMs, gene annotations (from paper's Google Drive)

data/reference/              Genome FASTAs (hg38, mm10), Ensembl GTFs, JASPAR motifs
data/processed/              Pipeline outputs (h5ad, CSV)
```

## Datasets

| | Retinal (mouse) | PBMC (human) | SEA-AD (human) |
|---|---|---|---|
| Type | Unpaired | Paired (10X Multiome) | Paired (multiome) + Unpaired (singleome) |
| Source | Paper's Seurat exports | 10X Genomics | Allen Institute / SEA-AD (MTG) |
| Donors / cells | 1 cohort / ~50k | 1 donor / ~10k | 28 paired + 84 unpaired / ~1.5M total |
| Cell types | RPC1/2/3, MG | 8 immune subsets | All 24 SEA-AD subclasses kept; subset per sub-question |
| Disease signal | dev timepoints | none | Full AD spectrum: ADNC, Braak, CERAD, APOE4, Cognitive Status |
| Phases 0-1 | Skipped (paper data) | Our pipeline | Our pipeline |

## Pipeline

### Phase 0-1: Data loading and integration

**Retinal:** We use the paper's pre-processed data directly. The Seurat objects
were exported to MatrixMarket format via R (see `scripts/run_paper_phase3.R`).
This gives us the exact same cells, cell types, and non-binary ATAC fragment
counts the paper used.

```bash
pixi run load-paper     # Load paper's retinal data -> data/processed/
```

**PBMC:** Processed from the 10X Multiome raw data using our Python pipeline.
Cell types are annotated with CellTypist (Python equivalent of Seurat's
reference-based label transfer).

```bash
pixi run process-pbmc       # Phase 0: load + annotate + QC
pixi run integrate-pbmc     # Phase 1: WNN integration
```

**SEA-AD:** Downloaded from the Allen Institute S3 bucket. Phase 0 splits
each h5ad by `method` (`10xMulti` = multiome / paired; otherwise singleome /
unpaired) and writes four files: `seaad_paired_{rna,atac}.h5ad` and
`seaad_unpaired_{rna,atac}.h5ad`. All 24 SEA-AD `Subclass` cell types are
kept on disk; subclass filtering is left to each sub-question.

Phase 1 has two branches. Paired uses the standard WNN pipeline. Unpaired
computes one global embedding (PCA + Harmony with `batch_key="modality"`)
across all 1.5M cells, then pairs RNA↔ATAC per-donor in that space. The
unpaired branch has known limitations (modality clouds remain visibly
separated; cell-type agreement is modest). For your the per-student RQs, use the paired data! It's documented in 
[docs/seaad_quickstart.md](docs/seaad_quickstart.md); for the full design
history and what's still open see
[docs/seaad_integration_history.md](docs/seaad_integration_history.md).

```bash
pixi run load-seaad           # Phase 0: split paired/unpaired
pixi run integrate-seaad      # Phase 1: WNN (paired) + global PCA/Harmony (unpaired)
pixi run inspect-seaad-integration   # QC figures + per-donor diagnostics
```

### Phase 2: Feature selection

Subsamples cells (100/type retinal, 50/type PBMC) and selects highly variable
genes (500 HVGs) and peaks (10,000 HVPs) using Seurat v3 VST.

Two modes are available:

- **Python mode** (default): uses scanpy's `seurat_v3` VST implementation.
  Produces 99.4% HVG overlap and identical correlation results compared to
  the R code (validated against `scripts/run_paper_phase3.R`).
- **R-reference mode**: uses pre-exported feature lists from Seurat's
  `FindVariableFeatures`. Gives an exact match with the R pipeline. Useful
  for validating the reproduction; not needed for new datasets.

```bash
pixi run feature-select         # Python mode (default, retinal + PBMC)
pixi run feature-select-r-ref   # R-reference mode (retinal only)
```

Students working on new datasets or modifying the feature selection step
should use the Python mode. The R-reference mode is provided to confirm
that differences between R and Python are limited to the VST
implementation, not the downstream logic.

**SEA-AD Phase 2 follows a different design.** SEA-AD has multi-donor
structure, so a pooled "50 cells per cell type" subsample mixes donors in
ways that break per-donor analyses. Instead, the SEA-AD Phase 2 script
picks HVGs/HVPs from the FULL ~138k-cell paired set (more stable VST
ranking) and leaves cell selection to each sub-question:

```bash
pixi run seaad-hvg-selection
```

Produces `seaad_paired_rna_hvg.h5ad` (full cells × 500 HVGs) and
`seaad_paired_atac_hvp.h5ad` (full cells × 10k HVPs), both carrying the
full SEA-AD obs, the joint WNN-input embedding in `obsm["X_pca"]`, and the
full-set k=20 WNN neighbor indices in `uns["wnn_neighbor_indices"]`.

A default Phase 3 input is pre-baked on the cluster via
`sbatch slurm/run_subsample_seaad_paired.sh` (no args ⇒ `--seed 42`, 50 cells
per all 24 subclasses → 1,200 cells, written as `seaad_paired_{rna,atac}_sub42.h5ad`).
Students who need a different selection rerun the wrapper with their own
arguments:

```bash
sbatch slurm/run_subsample_seaad_paired.sh --seed 7 --n-per-type 100    # bigger sample
sbatch slurm/run_subsample_seaad_paired.sh --seed 99 \
    --cell-types Microglia-PVM Astrocyte Oligodendrocyte                # glia only
```

The seed appears in the output filename (`seaad_paired_{rna,atac}_sub{seed}.h5ad`)
so multiple sub-questions can coexist; Phase 3 picks them all up via glob
and emits prefixed outputs (`seaad_paired_sub{seed}_*`).

### Phase 3: Gene-peak-TF relations

Establishes regulatory triplets (TF -> peak -> target gene):

1. Peak-gene overlap (250kb window around TSS)
2. Spearman correlation filtering (|r| > 0.1)
3. TF motif scanning in peaks (TRANSFAC motifs from the paper)
4. Triplet assembly (TF-motif-peak-gene joins)
5. RF input preparation (peak matrix + noise)

Uses the paper's exact TRANSFAC motif database and PWMs. Motif matching uses
MOODS (exact p-values, Linux/cluster) with a numpy fallback (approximate
thresholds, Windows).

Gene annotations: retinal uses the paper's `mouse.genes.gtf`; PBMC uses
`gencode.v38.annotation.gtf`; **SEA-AD uses `hg38.ensembl98.gtf.gz`** (same
as Phase 1 unpaired) — Ensembl 98 annotates all 500 SEA-AD HVGs including
the Ensembl-style novel-transcript IDs (AL/AC/AP-prefix) that GENCODE v38
silently drops (132/500 lost).

```bash
pixi run gene-peak
```

Phase 3 globs `seaad_{paired,unpaired}_{rna,atac}_sub*.h5ad` to pick up
every subsample present and emits prefixed outputs per run. Validate any
run's outputs with:

```bash
pixi run python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub42
```

### Validation summary

**Retinal benchmark.** Using R-reference features, the Python pipeline produces **identical results**
to the R code (Jaccard = 1.0 on correlated gene-peak pairs). Using Python
VST features:

| Step | R | Python | Agreement |
|---|---|---|---|
| HVGs | 500 | 500 | 497/500 (99.4%) |
| Peak-gene overlap | 1,298 | 1,341 | 1,337 shared |
| Correlated pairs | 172 | 228 | 153 shared (89% of R) |
| Correlation code | - | - | Identical (proven on shared features) |

The 6% difference in correlated pairs comes entirely from 3 different HVGs and
18 different HVPs selected by the LOESS implementations (R's Fortran `loess()`
vs Python's `skmisc.loess`). All downstream code (correlation, motif matching,
triplet assembly) is equivalent.

**SEA-AD paired (this work).** End-to-end Phase 3 run on a stratified
50-per-subclass subsample (1,200 cells × 24 subclasses) with Ensembl 98 GTF
and MOODS:

| Step | Count |
|---|---|
| HVGs annotated | 500/500 (vs 368/500 with GENCODE v38) |
| Peak-gene overlap pairs | 801 |
| Correlated pairs (\|r\|>0.1) | 421 |
| MOODS motif matches | 67,095 |
| Triplets | 10,424 |
| Unique (TF, target) candidate edges | ~5,200 |
| TFs / targets in candidates | 26 / 211 |

Numbers are at the same scale as the paper's retinal benchmark (~10k
triplets, ~25 TFs in the HVG set). The "26 TFs" ceiling is set by TRANSFAC
coverage and is the same for any human dataset using the paper's PWMs.

## Output: inputs for network inference (Fig. 2, step 3+)

The preprocessing pipeline (Phases 0-3) produces everything needed for
cell-specific network inference. After running the pipeline, these files
are in `data/processed/`:

| File | Shape | Contents |
|---|---|---|
| `*_rna_sub.h5ad` | (n, 500) | Subsampled RNA, 500 HVGs, raw counts |
| `*_atac_sub.h5ad` | (n, 10000) | Subsampled ATAC, 10k peaks, raw counts |
| `*_knn_indices.npy` | (n, 20) | KNN neighbor indices from integrated embedding |
| `*_triplets.csv` | ~10k rows | (TF, peak, target_gene) regulatory triplets |
| `*_gene_labels.csv` | 500 rows | Each gene labeled as TF or target |
| `*_peak_gene_pairs.csv` | ~hundreds | Correlated gene-peak pairs with Spearman r |
| `*_peak_overlap_matrix.npz` | (n, ~hundreds) | Peak accessibility + noise, for RF input |
| `*_peak_info.csv` | ~hundreds rows | Which peaks map to which genes |

Where `*` is `retinal`, `pbmc`, or for SEA-AD `seaad_paired_sub{seed}` /
`seaad_unpaired_sub{seed}` (seed = the random seed used in
[`scripts/subsample_seaad_paired.py`](scripts/subsample_seaad_paired.py)).
For retinal/PBMC `n=400`; for SEA-AD `n` is whatever the student picked.

The SEA-AD HVG/HVP files produced by Phase 2 additionally carry
`obsm["X_pca"]` (joint WNN-input embedding, 40 dims) and
`uns["wnn_neighbor_indices"]` (full-set k=20 WNN KNN). Students should
recompute KNN on the subsample from `obsm["X_pca"]` rather than using the
full-set indices directly:

```python
from sklearn.neighbors import NearestNeighbors
import anndata as ad
rna = ad.read_h5ad("data/processed/seaad/seaad_paired_rna_sub42.h5ad")
knn = NearestNeighbors(n_neighbors=20).fit(rna.obsm["X_pca"])
_, indices = knn.kneighbors(rna.obsm["X_pca"])   # subsample-local KNN
```

**To load in Python:**

```python
import anndata as ad
import pandas as pd
import numpy as np

# Subsampled expression + accessibility
rna = ad.read_h5ad("data/processed/retinal_rna_sub.h5ad")   # (400, 500)
atac = ad.read_h5ad("data/processed/retinal_atac_sub.h5ad")  # (400, 10000)

# KNN neighbor indices (from Harmony embedding, k=20)
knn = np.load("data/processed/retinal_knn_indices.npy")      # (400, 20)

# Regulatory triplets: which TFs regulate which genes via which peaks
triplets = pd.read_csv("data/processed/retinal_triplets.csv")
# Columns: TF, peak, target_gene, spearman_r

# Gene labels: TF or target
labels = pd.read_csv("data/processed/retinal_gene_labels.csv")

# Peak matrix for RF input (accessibility + Gaussian noise)
peak_data = np.load("data/processed/retinal_peak_overlap_matrix.npz")
peak_matrix = peak_data["peak_matrix"]  # (400 cells, ~200 peaks)
```

The triplets table and gene labels are the key inputs to the network
inference step (wScReNI, kScReNI, etc. in Figure 2 of the paper). Each
student's sub-question builds on these files.

## Running on the cluster

```bash
# Retinal
pixi run load-paper
pixi run feature-select
pixi run gene-peak

# PBMC
pixi run process-pbmc
pixi run integrate-pbmc
pixi run feature-select
pixi run gene-peak

# SEA-AD (Phase 0 + 1) — one-off
pixi run load-seaad
pixi run integrate-seaad
pixi run inspect-seaad-integration       # QC: per-donor UMAPs + donor summary

# SEA-AD HVG/HVP selection (Phase 2 equivalent) — one-off
pixi run seaad-hvg-selection

# SEA-AD default subsample (50/type × 24 subclasses = 1,200 cells, seed=42) — one-off
sbatch slurm/run_subsample_seaad_paired.sh

# SEA-AD Phase 3 — globs every _sub*.h5ad and emits one set of outputs per run
sbatch slurm/run_gene_peak.sh
pixi run python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub42
```

For sub-questions that need a different cell selection, students rerun
`sbatch slurm/run_subsample_seaad_paired.sh --seed 123` (and optionally
`--n-per-type` / `--cell-types`) and then `sbatch slurm/run_gene_peak.sh`
picks the new files up alongside the default `_sub42` files. See
[docs/seaad_quickstart.md](docs/seaad_quickstart.md).

Container-side check that MOODS imports cleanly (avoids silent fallback to
the approximate numpy scanner):

```bash
bash scripts/check_moods_in_container.sh
```

## Dependencies

All pure Python -- no R/rpy2 required at runtime. R is only used for
one-time data export (`scripts/run_paper_phase3.R`) and validation.

| Package | Purpose |
|---|---|
| scanpy | scRNA-seq preprocessing, HVG selection |
| anndata | Data containers |
| muon | WNN integration, ATAC TF-IDF/LSI |
| harmonypy | Batch correction |
| celltypist | PBMC cell type annotation |
| pyfaidx | Genome FASTA reading |
| rdata | Read R .rds files (TRANSFAC PWMs) |
| MOODS-python | Exact motif p-value matching (Linux only) |

