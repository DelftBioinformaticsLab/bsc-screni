

## Retinal Development in Mice

ScReNI paper says:
> The unpaired scRNA-seq and scATAC-seq data were publicly available from the study of retinal development in mice, which can be accessed at Gene Expression Omnibus (GEO: GSE181251) [25]. The original research reported 13 major retinal cell types. For this study, we focused on three subtypes of retinal progenitor cells (RPCs), designated as RPC1, RPC2, and RPC3, as well as Müller glial (MG) cells, utilizing their original cell type annotations.

--> we downloaded everything from the that GEO page to `data/retinal_GSE181251`

## PMBC 10X

ScReNI paper says:
> The paired scRNA-seq and scATAC-seq data were obtained from PBMCs of a healthy donor which were obtained by 10X Genomics (https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-2-0-0). The libraries for paired gene expression and chromatin accessibility were generated from the isolated nuclei as described in the Chromium Next GEM Single Cell Multiome ATAC + Gene Expression User Guide (CG000338 Rev A) and sequenced on Illumina NovaSeq 6000 v1 Kit (Forward Strand Dual-Indexed Workflow, Illumina, San Diego, CA). For this research, we selected single-cell sequencing data from CD14 monocytes, CD16 monocytes, CD4 naive cells, CD8 naive cells, conventional dendritic cells (cDCs), memory B cells, natural killer (NK) cells, and regulatory T (Treg) cells.

--> we downloaded **only the relevant files** from the link mentioned above, into `data/pbmc_unsorted_10k/`
