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

Pixi should already be available on the cluster.
Verify the installation with:

```bash
pixi --version
```

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

For cluster jobs, use a/Apptainer container instead of activating pixi directly. This avoids environment activation issues inside SLURM and gives fully reproducible runs.

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

### Useful commands

| Command | Description |
|---|---|
| `pixi install` | Install / sync the locked environment |
| `pixi run <task>` | Run a named pipeline task |
| `pixi shell` | Activate an interactive shell in the environment |
| `pixi add <package>` | Add a new dependency to `pixi.toml` |
| `pixi list` | List installed packages |

### Troubleshooting

- If `pixi` is not found after installation, run `source ~/.bashrc` or log out and back in.
- If you get a lock conflict, delete `.pixi/` and re-run `pixi install`.
- If the premade `.sif` is missing or outdated, build your own with `apptainer build`.

---

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
| Type | Unpaired | Paired (10X Multiome) | Both |
| Source | Paper's Seurat exports | 10X Genomics | Allen Institute (AWS) |
| Cell types | RPC1, RPC2, RPC3, MG | 8 immune types | Micro, Astro, Oligo, L2/3 IT |
| Phases 0-1 | Skipped (use paper data) | Our pipeline | Our pipeline |

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
pixi run feature-select         # Python mode (default)
pixi run feature-select-r-ref   # R-reference mode (retinal only)
```

Students working on new datasets or modifying the feature selection step
should use the Python mode. The R-reference mode is provided to confirm
that differences between R and Python are limited to the VST
implementation, not the downstream logic.

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

```bash
pixi run gene-peak
```

### Validation summary (retinal benchmark)

Using R-reference features, the Python pipeline produces **identical results**
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

## Output: inputs for network inference (Fig. 2, step 3+)

The preprocessing pipeline (Phases 0-3) produces everything needed for
cell-specific network inference. After running the pipeline, these files
are in `data/processed/`:

| File | Shape | Contents |
|---|---|---|
| `*_rna_sub.h5ad` | (400, 500) | Subsampled RNA, 500 HVGs, raw counts |
| `*_atac_sub.h5ad` | (400, 10000) | Subsampled ATAC, 10k peaks, raw counts |
| `*_knn_indices.npy` | (400, 20) | KNN neighbor indices from integrated embedding |
| `*_triplets.csv` | ~10k rows | (TF, peak, target_gene) regulatory triplets |
| `*_gene_labels.csv` | 500 rows | Each gene labeled as TF or target |
| `*_peak_gene_pairs.csv` | ~200 rows | Correlated gene-peak pairs with Spearman r |
| `*_peak_overlap_matrix.npz` | (400, ~200) | Peak accessibility + noise, for RF input |
| `*_peak_info.csv` | ~200 rows | Which peaks map to which genes |

Where `*` is `retinal`, `pbmc`, or `seaad_paired`/`seaad_unpaired`.

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

# SEA-AD
pixi run load-seaad
pixi run integrate-seaad
pixi run feature-select
pixi run gene-peak
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

