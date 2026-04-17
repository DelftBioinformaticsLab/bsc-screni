# Data Processing Pipeline

Python reimplementation of the ScReNI preprocessing pipeline (Xu et al. 2025,
GPB). Code lives in `src/screni/data/`. Students implement the algorithmic
phases (network inference, evaluation, clustering, regulators) separately.

## Datasets

| | Retinal (mouse) | PBMC (human) | SEA-AD (human) |
|---|---|---|---|
| Type | Unpaired | Paired (10X Multiome) | Both (paired + unpaired) |
| Source | Paper's Seurat exports | 10X Genomics | Allen Institute (AWS) |
| Cell types | RPC1, RPC2, RPC3, MG | CD14 mono, CD16 mono, CD4 naive, CD8 naive, cDC, Memory B, NK, Treg | Micro-PVM, Astro, Oligo, L2/3 IT |

---

## Phase 0: Data Loading

### Retinal (`loading_paper.py`)

We use the paper's pre-processed Seurat objects, exported to MatrixMarket
format via R. This gives us the exact same data the paper used:

- **RNA**: 27,933 genes x 48,377 cells (RPCs_S1/S2/S3 + MG, raw counts)
- **ATAC**: 284,850 peaks x 29,193 cells (non-binary fragment counts, max=257)
- **Cell annotation**: 400 subsampled cells (100/type) with matched RNA-ATAC
  pairs from the paper's `Cell100_annotation.csv`

The GEO deposit (Lyu et al. 2021, GSE181251) contains binarized ATAC data
from different peak calling. The paper's Seurat objects have real fragment
counts from Signac processing, which is required for the correlation step.

Command: `pixi run load-paper`

### PBMC (`loading.py`)

Processed from the 10X Multiome raw data (PBMC Unsorted 10k).

The ScReNI paper provides no PBMC annotation code. They used Seurat
reference-based label transfer (Hao et al. 2021 reference). We use CellTypist
with the `Immune_All_Low.pkl` model as the Python equivalent, mapping
fine-grained labels to the paper's 8 types.

QC filtering: `n_genes_by_counts > 200`, `< 4500`, `pct_counts_mt < 15`.

Command: `pixi run process-pbmc`

### SEA-AD (`loading_seaad.py`)

Downloads ~30 GB h5ad files from AWS S3, filters to 4 AD-relevant cell types,
splits into paired (multiome) and unpaired (singleome) branches.

Command: `pixi run load-seaad`

---

## Phase 1: Integration

### PBMC -- Paired (`integration.py`)

WNN integration (same algorithm as Seurat v4):
1. RNA: normalize -> log1p -> HVG(2000) -> scale -> PCA(50)
2. ATAC: TF-IDF -> LSI(50)
3. L2-normalize embeddings, per-modality neighbor graphs
4. WNN via muon, Leiden clustering, joint UMAP

Command: `pixi run integrate-pbmc`

### Retinal

**Skipped.** The paper's exported data already contains the final cell type
labels and matched RNA-ATAC pairs.

### SEA-AD (`integration_seaad.py`)

Two branches:
- **Paired**: WNN with donor-aware HVG selection
- **Unpaired**: per-donor Harmony alignment, cross-modal NN pairing

Command: `pixi run integrate-seaad`

---

## Phase 2: Feature Selection (`feature_selection.py`)

Matches `select_features()` and `Select_partial_cells_for_scNewtorks()` from
the R code.

**Steps:**
1. Subsample cells: 100/type (retinal) or 50/type (PBMC, SEA-AD)
2. For unpaired data: subsample matched RNA-ATAC pairs to keep alignment
3. Filter ATAC to chr-prefixed peaks
4. Select 500 HVGs via Seurat v3 VST (returns raw counts)
5. Select 10,000 HV peaks via Seurat v3 VST (returns raw counts)

### Python vs R feature selection

Two modes are available:

**Python mode** (default, `pixi run feature-select`): uses scanpy's `seurat_v3`
VST. Validated against the R code on the retinal benchmark:
- 497/500 HVGs match (99.4%)
- 18/10,000 HVPs differ
- All differences are from the LOESS regression implementation (R's Fortran
  `loess()` vs Python's `skmisc.loess()`); the ranking algorithm is identical

**R-reference mode** (`pixi run feature-select-r-ref`): uses pre-exported
feature lists from R's `FindVariableFeatures`. Produces an exact match
(Jaccard = 1.0 on all downstream results). The feature lists are generated
by `scripts/run_paper_phase3.R`.

Students should use Python mode for new datasets or when modifying feature
selection. R-reference mode is for paper reproduction validation only.

---

## Phase 3: Gene-Peak-TF Relations (`gene_peak_relations.py`)

Matches `Infer_gene_peak_relationships()` from the R code. Uses the paper's
exact TRANSFAC motif database and PWMs.

### 3a. Peak-gene overlap

Find peaks within 250kb of each gene's TSS using the paper's GTF
(`mouse.genes.gtf` for retinal, `gencode.v38.annotation.gtf` for human).

Strand-aware: upstream/downstream are swapped for minus-strand genes
(matching R's `get_tss_region()`).

### 3b. Correlation filtering

Spearman correlation between gene expression and peak accessibility on cells
where both are non-zero. Keep pairs with |r| > 0.1. Matches R's
`gene_peak_corr1()` exactly (validated: Jaccard = 1.0 on shared features).

### 3c. Motif matching

Scan correlated peaks (not all peaks) for TF binding motifs:
- **Motifs**: paper's TRANSFAC database (`Tranfac201803_*_MotifTFsFinal`)
- **PWMs**: paper's `all_motif_pwm.rds` (loaded via `rdata` package)
- **Pre-filtering**: only scan motifs whose TFs are in the gene set
  (matching R's `motifs_select()`)
- **Engine**: MOODS for exact p-values (Linux); numpy fallback with
  normal-approximation thresholds (Windows)

### 3d. Triplet assembly

Join motif-peak matches with correlated gene-peak pairs via the TRANSFAC
motif-TF mapping (matching R's `peak_gene_TF_match()`). Remove
self-regulation.

### 3e. RF input preparation

Extract peak accessibility matrix for correlated peaks, add Gaussian noise
N(0, 10^-5), label genes as TF or target.

Command: `pixi run gene-peak`

### Validation (retinal benchmark)

With R-reference features (exact same HVGs + HVPs):

| Metric | R | Python | Match |
|---|---|---|---|
| Overlap pairs | 1,298 | 1,298 | 100% |
| Correlated pairs | 172 | 172 | 100% |
| Jaccard | - | - | 1.000 |

With Python VST features (independent feature selection):

| Metric | R | Python | Match |
|---|---|---|---|
| HVGs | 500 | 500 | 497 shared (99.4%) |
| Overlap pairs | 1,298 | 1,341 | 1,337 shared |
| Correlated pairs | 172 | 228 | 153 shared (Jaccard 0.62) |

The 38% Jaccard gap comes entirely from different feature selection at the
margins. The correlation and downstream code is identical.

---

## File structure

```
src/screni/data/
    loading.py               Phase 0 PBMC: load + CellTypist + QC
    loading_paper.py         Phase 0 retinal: load paper's Seurat exports
    loading_seaad.py         Phase 0 SEA-AD: load + split paired/unpaired
    integration.py           Phase 1 PBMC + SEA-AD paired: WNN
    integration_seaad.py     Phase 1 SEA-AD: WNN + per-donor Harmony
    feature_selection.py     Phase 2: subsample + HVG/HVP selection
    gene_peak_relations.py   Phase 3: peak-gene overlap, correlation, motifs
    utils.py                 Peak parsing, GTF loading, gene activity

scripts/
    run_paper_phase3.R       Validation: run paper's R code, export features

data/paper/
    datasets/                Paper's exported data (Seurat -> MTX/CSV)
    reference/               TRANSFAC motifs, PWMs, GTFs
```

## Dependencies

All pure Python -- no R/rpy2 at runtime.

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
