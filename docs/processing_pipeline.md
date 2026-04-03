# Data Processing Pipeline

Python reimplementation of the ScReNI preprocessing pipeline (Xu et al. 2025,
GPB). This document describes all choices made and how they compare to the
original R implementation.

Code lives in `src/screni/data/`. Students implement the algorithmic phases
(network inference, evaluation, clustering, regulators) separately.

## Datasets

| | Retinal (mouse) | PBMC (human) |
|---|---|---|
| Type | Unpaired | Paired (10X Multiome) |
| RNA source | Clark et al. 2019 (GSE118614) | 10X Genomics |
| ATAC source | Lyu et al. 2021 (GSE181251) | Same cells |
| Cell types | RPC1, RPC2, RPC3, MG | CD14 mono, CD16 mono, CD4 naive, CD8 naive, cDC, Memory B, NK, Treg |

---

## Phase 0: Data Loading (`loading.py`)

### Retinal

**Timepoint selection.** The paper states "age-matched nine time points" and
Figure 2A confirms: E11, E12, E14, E16, E18, P0, P2, P5, P8. We exclude P14
(present in both RNA and ATAC datasets) because the paper does. P14 contains
mostly mature MG cells (937/940 cells at P14 are MG), and excluding it brings
our MG count from 1710 to 773 (paper: 768).

**Cell type names.** The ATAC data uses `RPCs_S1/S2/S3/MG` which we map to
`RPC1/RPC2/RPC3/MG`. The RNA data uses Clark 2019 labels (`Early RPCs`,
`Late RPCs`, `Muller Glia`, etc.) which are NOT the final ScReNI labels.
Final RNA labels are assigned after integration (Phase 1) by transferring
ATAC labels via nearest-neighbor matching.

**RNA pre-filtering.** We coarsely filter RNA cells to `Early RPCs`,
`Late RPCs`, `Muller Glia`, and `Neurogenic Cells` before integration to
reduce computation. This is a superset of what ends up as RPC1/2/3/MG.

**Retinal cell count comparison (ATAC, after filtering):**

| Type | Ours | Paper | Notes |
|---|---|---|---|
| RPC1 | 6736 | 6049 | +11%, will shift after integration |
| RPC2 | 10966 | 10464 | +5% |
| RPC3 | 11966 | 11912 | ~exact |
| MG | 773 | 768 | ~exact |

The remaining RPC1/RPC2 discrepancy is expected to resolve during Phase 1
integration, where Seurat's CCA anchor transfer can reassign borderline cells.

### PBMC

**Cell type annotation.** The ScReNI paper provides no annotation code. They
used Seurat reference-based label transfer (FindTransferAnchors + TransferData
against the Hao et al. 2021 PBMC reference). We use CellTypist with the
`Immune_All_Low.pkl` model as the Python equivalent.

CellTypist produces fine-grained labels (17 types on this dataset) which we
map to the paper's 8 types:

| CellTypist label | ScReNI type |
|---|---|
| Classical monocytes | CD14 monocyte |
| Non-classical monocytes | CD16 monocyte |
| Tcm/Naive helper T cells | CD4 naive cell |
| Tcm/Naive cytotoxic T cells | CD8 naive cell |
| DC2 | cDC |
| Memory B cells | Memory B cell |
| CD16+ NK cells | NK |
| Regulatory T cells | Treg |

Unmapped subtypes (Tem/Temra, MAIT, Naive B, pDC, Plasma cells, etc.) are
dropped. Both the full (all CellTypist labels) and filtered (8 types) h5ad
files are saved.

**QC filtering.** Applied before saving:
- `n_genes_by_counts > 200`
- `n_genes_by_counts < 4500`
- `pct_counts_mt < 15`

This removes ~2600 cells (22%).

**PBMC cell count comparison (after QC + mapping):**

| Type | Ours | Paper | Notes |
|---|---|---|---|
| CD14 mono | 3316 | 2812 | +18%, CellTypist broader |
| CD16 mono | 349 | 514 | -32% |
| CD4 naive | 2452 | 1419 | +73%, includes central memory |
| CD8 naive | 699 | 1410 | -50%, CellTypist more restrictive |
| cDC | 74 | 198 | -63% |
| Memory B | 308 | 371 | -17% |
| NK | 407 | 468 | -13% |
| Treg | 137 | 162 | -15% |

NK, Memory B, and Treg are close. T cell and monocyte differences are from
CellTypist vs Seurat classifying these populations differently.

### Diagnostic outputs

Phase 0 generates plots in `output/data_inspection/`:
- `pbmc_qc_violin.png` — QC metrics per CellTypist cell type
- `pbmc_celltypes_umap.png` — 3-panel UMAP: RNA ScReNI types, RNA CellTypist
  labels, ATAC LSI embedding

---

## Phase 1: Integration

### PBMC — Paired (`integration.py`)

Follows `Integrate_scRNA_scATAC(data.type='paired')` from the R code.

**Steps:**
1. RNA: `normalize_total(1e4)` → `log1p` → `HVG(2000, seurat_v3, layer=counts)` → `scale` → `PCA(50)`
2. ATAC: filter zero-count peaks → `TF-IDF` → `LSI(50)` (component 1 kept for WNN, dropped for standalone UMAP)
3. L2-normalize PCA and LSI embeddings (matching Seurat's default before WNN)
4. Per-modality neighbor graphs (`n_pcs=20`, matching `IntegratedDimensions`)
5. WNN via `muon.pp.neighbors()` (same algorithm as Seurat v4)
6. Leiden clustering (`resolution=0.8`, matching Seurat default)
7. Joint UMAP from WNN graph
8. Extract WNN neighbor indices `(n_cells, k=20)` for downstream ScReNI

**Differences from R pipeline:**

| Step | Paper (R) | Ours (Python) | Impact |
|---|---|---|---|
| RNA normalization | SCTransform | normalize + log1p | Minor — WNN is adaptive |
| Clustering | SLM (algorithm 3) | Leiden | Minor — very similar |
| LSI component 1 | Included in WNN | Included | Same |
| L2 normalization | Seurat default | Explicit `sklearn.preprocessing.normalize` | Same |

**Execution status:** Ran locally (Windows). 13 clusters, validation passed.
See `docs/pbmc_integration_review.md` for detailed validation results.

**Output:** `pbmc_integrated.h5mu` (MuData with WNN graph, UMAP, clusters,
neighbor indices in `uns['wnn_neighbor_indices']`), `output/integration/pbmc_wnn_umap.png`.

### Retinal — Unpaired (`integration_retinal.py`)

Follows `Integrate_scRNA_scATAC(data.type='unpaired')` from the R code.

**Steps:**
0. Gene activity: count peaks overlapping gene body + 2kb upstream (Signac
   `GeneActivity` equivalent). Uses sparse matrix multiplication for speed.
   Also compute LSI on raw ATAC peaks.
1. RNA: `normalize_total(1e4)` → `log1p` → `HVG(2000, seurat_v3, layer=counts)` → `scale`
2. Gene activity: `normalize_total(1e4)` → `log1p` → `scale`
3. CCA anchor finding: SVD on RNA gene space, project ATAC gene activity into
   same space, L2-normalize, find mutual nearest neighbors (k=5)
4. Impute RNA expression onto ATAC cells: weighted average of anchor RNA cells'
   expression. ATAC cells without direct anchors get imputed from nearest
   anchored ATAC cell in LSI space.
5. Merge RNA + imputed ATAC on shared HVGs
6. PCA on merged (center only, `do.scale=FALSE` matching Seurat)
7. Harmony (`lambda=0.5`, dims 2:20, batch=datatype)
8. UMAP from Harmony embedding
9. Neighbors (k=20) from Harmony embedding
10. Cross-modality pairing: Euclidean nearest neighbor in 2D UMAP space,
    deduplicate (1:1 matching), transfer cell type labels from ATAC → RNA

**Differences from R pipeline:**

| Step | Paper (R) | Ours (Python) | Impact |
|---|---|---|---|
| Gene activity | Signac (fragment-based) | Peak overlap counting | Similar results |
| CCA | Seurat `FindTransferAnchors` | SVD on RNA space + MNN | Approximate — uses RNA PCA projection instead of true cross-covariance CCA |
| Imputation | Seurat `TransferData(weight.reduction=lsi)` | Anchor-weighted averaging | Similar concept, simplified weighting |
| Harmony `reduction.use` | Code says 'lsi' (likely bug) | PCA | More standard |

**Cross-modality pairing results:**

| Type | Paired cells | Paper RNA count |
|---|---|---|
| RPC1 | 4,956 | 7,853 |
| RPC2 | 7,837 | 16,645 |
| RPC3 | 8,991 | 22,943 |
| MG | 628 | 936 |
| **Total** | **22,412** | **48,377** |

The 1:1 deduplication is inherently lossy (62k RNA cells compete for 30k ATAC
cells). All types have >100 paired cells, which is sufficient for ScReNI's
downstream subsampling (100 cells per type).

---

## Phase 2: Feature Selection (`feature_selection.py`)

*Implemented but not yet tested.*

Matches `select_features()` and `Select_partial_cells_for_scNewtorks()` from
the R code:
- Subsample 100 cells per cell type (retinal) or 50 per type (PBMC)
- Select 500 HVGs (seurat_v3 VST) — returns raw counts
- Select 10,000 HV peaks — returns raw counts
- Filter ATAC peaks to chr-prefixed only

---

## Phase 3: Gene-Peak-TF Relations (`gene_peak_relations.py`)

*Implemented but not yet tested.*

Matches `peak_gene_overlap_GR1()`, `gene_peak_corr1()`,
`Get_motif_peak_pair_df0()`, and `peak_gene_TF_match()` from the R code:

1. **Peak-gene overlap**: Find peaks within 250kb of each gene's TSS
   (pandas interval intersection, no pyranges dependency)
2. **Correlation filtering**: Spearman correlation between gene expression and
   peak accessibility, keep |r| > 0.1
3. **Motif matching**: Pure numpy PWM scanner (no gimmemotifs/MOODS dependency).
   Scans peak sequences against JASPAR vertebrate motifs, p-value < 5^-4
4. **Triplet assembly**: Inner join of correlated gene-peak pairs with
   TF motif-peak matches → (TF, peak, target_gene) triplets
5. **RF input preparation**: Extract peak overlap matrix with Gaussian noise
   N(0, 10^-5), label genes as TF or target

---

## Dependencies

All pure Python — no R/rpy2.

| Package | Purpose |
|---|---|
| scanpy | Core scRNA-seq preprocessing |
| anndata | Data containers |
| muon | WNN integration, ATAC TF-IDF/LSI |
| harmonypy | Batch correction (same algorithm as R) |
| celltypist | PBMC cell type annotation |
| scikit-learn | CCA (SVD), nearest neighbors, clustering |
| pyfaidx | Genome FASTA reading (motif matching, Linux only) |

---

## File structure

```
src/screni/data/
    __init__.py
    loading.py               # Phase 0: load + QC + annotate + save
    integration.py           # Phase 1 PBMC: WNN integration
    integration_retinal.py   # Phase 1 retinal: CCA + Harmony integration
    feature_selection.py     # Phase 2: subsample + HVG/HVP selection
    gene_peak_relations.py   # Phase 3: peak-gene overlap, correlation, motifs
    utils.py                 # Peak name parsing

src/
    inspect_data.py          # Dataset inspection + stats
    plot_integration.py      # Integration diagnostic UMAPs

slurm/
    download_pbmc.sh         # Download PBMC 10X Multiome
    download_retinal.sh      # Download retinal RNA + ATAC
    download_reference.sh    # Download mm10/hg38 GTFs + FASTAs + JASPAR
    run_processing.sh        # Run Phase 0 in container
    inspect_data.sh          # Run inspection in container
```
