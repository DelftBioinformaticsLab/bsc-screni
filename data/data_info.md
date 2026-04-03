

## Retinal Development in Mice

The ScReNI paper uses unpaired scRNA-seq and scATAC-seq from mouse retinal development. These come from two separate studies by the same lab (Blackshaw lab, Johns Hopkins).

### scATAC-seq — GSE181251 (Lyu et al. 2021, Cell Reports)

> The unpaired scRNA-seq and scATAC-seq data were publicly available from the study of retinal development in mice, which can be accessed at Gene Expression Omnibus (GEO: GSE181251) [25]. The original research reported 13 major retinal cell types. For this study, we focused on three subtypes of retinal progenitor cells (RPCs), designated as RPC1, RPC2, and RPC3, as well as Müller glial (MG) cells, utilizing their original cell type annotations.

Downloaded to `data/retinal_GSE181251/`. Key files:
- `Single_Cell_ATACseq_raw_matrix.mtx.gz` — 283,847 peaks x 94,318 cells (98.6% sparse)
- `Single_Cell_ATACseq_cell_annotation.txt.gz` — cell types (`celltypes` column, 13 types), timepoints (`realtime`, 11 timepoints E11–P14), UMAP coords
- `Single_Cell_ATACseq_peak_annotation.txt.gz` — peak annotations (note: column header is malformed, needs fixing)
- Plus perturbation experiment files (overexpression, knockout)

Cell type names in ATAC data: `RPCs_S1`, `RPCs_S2`, `RPCs_S3`, `MG`, `Rod`, `BC`, `AC_HC`, `RGC`, `Early_Rod`, `Early_NG`, `Late_NG`, `Early_Cone`, `Cone`

### scRNA-seq — GSE118614 (Clark et al. 2019, Neuron)

The developmental scRNA-seq is NOT in GSE181251. It comes from an earlier study by the same lab:
- **Paper**: Clark et al. 2019, "Single cell RNA-Seq analysis of retinal development identifies NFI factors as regulating mitotic exit and late-born cell specification", *Neuron*
- **GEO**: GSE118614 (raw data only, no annotations)
- **Processed data**: https://github.com/gofflab/developing_mouse_retina_scRNASeq (files hosted on Dropbox)

Downloaded to `data/retinal_GSE181251/scRNAseq_clark2019/`. Files:
- `10x_mouse_retina_development.mtx` — count matrix, 120,804 cells x 10 timepoints (E11–P14)
- `10x_mouse_retina_pData_umap2_CellType_annot_w_horiz.csv` — cell annotations (updated 2021, includes horizontal cells). Columns: `barcode`, `age`, `CellType`, `umap_CellType`, UMAP coords, etc.
- `10x_mouse_retina_development_phenoData.csv` — original cell annotations
- `10x_mouse_retina_development_featureData.csv` — gene annotations (Ensembl IDs, gene names)

Cell type names in RNA data differ from ATAC: `RPCs`, `Early RPCs`, `Late RPCs`, `Müller Glia`, `Rods`, `Cones`, `Amacrine`, `Bipolar`, `RGCs`, `Neurogenic`, `Photo. Precurs.`, `Horizontal`

### Mapping between the two datasets

The ScReNI paper uses "age-matched nine time points" (confirmed from Figure 2A
UMAP legend): **E11, E12, E14, E16, E18, P0, P2, P5, P8**.

- P11 is excluded because the RNA dataset (Clark 2019) doesn't have it.
- P14 is excluded by the authors' choice — it contains mostly mature MG cells
  (937/940 cells at P14 are MG). Dropping P14 brings the MG count from 1710 to
  773, matching the paper's 768.

The cell type names differ between the two datasets: ATAC uses `RPCs_S1/S2/S3`
and `MG`, while RNA uses `Early RPCs`, `Late RPCs`, `Muller Glia`, etc. The
final RPC1/2/3/MG labels are assigned to RNA cells after cross-modal
integration by transferring labels from matched ATAC cells.

## PBMC 10X

ScReNI paper says:
> The paired scRNA-seq and scATAC-seq data were obtained from PBMCs of a healthy donor which were obtained by 10X Genomics (https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-2-0-0). The libraries for paired gene expression and chromatin accessibility were generated from the isolated nuclei as described in the Chromium Next GEM Single Cell Multiome ATAC + Gene Expression User Guide (CG000338 Rev A) and sequenced on Illumina NovaSeq 6000 v1 Kit (Forward Strand Dual-Indexed Workflow, Illumina, San Diego, CA). For this research, we selected single-cell sequencing data from CD14 monocytes, CD16 monocytes, CD4 naive cells, CD8 naive cells, conventional dendritic cells (cDCs), memory B cells, natural killer (NK) cells, and regulatory T (Treg) cells.

Downloaded **only the relevant files** to `data/pbmc_unsorted_10k/`:
- `filtered_feature_bc_matrix.h5` — 12,012 cells x 148,458 features (36,601 genes + 111,857 ATAC peaks)
- `atac_fragments.tsv.gz` + `.tbi` — per-fragment chromatin accessibility
- `atac_peaks.bed` — 111,908 peak locations
- `atac_peak_annotation.tsv` — peak-gene associations (distal/promoter/intergenic)
- Metadata: `summary.csv`, `per_barcode_metrics.csv`, `web_summary.html`, `analysis.tar.gz`
- Skipped: BAMs (94 GB), CLOUPE, BIGWIG — not needed for scReNI
