# Using Cell-specific Gene Regulatory Networks to Understand Alzheimer’s Disease

**Abstract**: Alzheimer’s disease is a complex brain disorder that is still not fully understood. New single-cell sequencing technologies let us measure gene activity for thousands of individuals cells, resulting in large, high-dimensional datasets. This enables data-driven discovery of gene regulatory networks (GRNs), which describe how genes influence each other. In this project, you’ll implement and extend a recent machine-learning-based method called scReNI, which constructs cell-specific GRNs by integrating multiple types of single-cell data. You will apply it to a large Alzheimer’s dataset and explore technical and data-driven research questions. The project is computationally focused and combining machine learning and graph-based modeling with a real-world biomedical application.

_CS focus_: high-dimensional modeling; graph inference; graph algorithms; multi-modal integration; robustness; scalability.

_Prerequisites_: Programming in Python (required), Programming in R (useful), Machine Learning course (highly recommended), Interest in biology or bioinformatics (not strictly required, but makes the project more fun)


## Background and motivation
Alzheimer’s disease (AD) is the most common form of dementia, affecting millions of people worldwide. While research has uncovered the core processes that underlie the disease-such as the aggregation of plaques, and the formation of neurofibrillary tangles- current treatments only partially alleviate symptoms, and a definite cure for AD is yet to be discovered[1,2]. A more thorough understanding of the disease processes at a cellular or molecular level could be valuable in this search. In this project, you will have the opportunity to contribute to the data analysis methods that are used in this type of research.

From a Computer Science perspective, this challenge translates into analyzing complex, noisy, and high-dimensional biological data. Modern single-cell sequencing experiments measure gene expression levels for thousands of genes across tens of thousands of cells, producing large tabular datasets that require scalable and robust computational methods. One powerful way to analyze such data is by gene regulatory networks (GRNs)[3]. A GRN can be represented as a graph, where nodes correspond to genes and edges represent inferred regulatory relationships. More recent approached go one step further by constructing a separate GRN for each individual cell, enabling comparison of regulatory mechanisms across different cell types or disease states.

A recent method called scReNI[4] introduces a machine-learning-based framework to construct these cell-specific GRNs by integrating multiple single-cell data modalities into a single model. This makes it possible to study gene regulation at a unprecedented resolution.

## Project Description
In scReNI, gene expression dynamics are modeled using random forest regression models. Feature importance scores learned by the model are then interpreted as regulatory influences and translated into a graph structure representing a GRN. Although scReNI is a promising technique, it has not been extensively applied to disease-focused datasets. In this project you will apply it to SeaAD, a large publicly available single-cell Alzheimer’s disease dataset. Because the method is very recent, there are many open computational questions to be explored in this project.

This project is computational in nature, with biological interpretation providing real-world relevance, rather than being the main focus. You will work with:
- Large, high-dimensional tabular data
- Machine learning models (random forests)
- Graph representations and graph-based analysis
During the first weeks of the project, you will work together to understand and implement the scReNI model. After that, you will all define and investigate your own sub-question using this framework.

## Research Questions for the Sub-Projects
Each sub-project focuses on a well-defined computational research question within the broader scReNI framework. Students will work with machine learning models, graph representations, and large-scale single-cell datasets, while using Alzheimer’s disease as an application domain. For each sub-project, relevant literature is cited.

1. GRN association with AD: How can we compare regulatory networks, in order to identify parts (edges, or maybe even entire modules), that change because of the disease[5,6]? Does this effect change between cell types?

2. GRN stability: The random forest used to model the importance of other genes may have lower accuracy for different cell types or outliers. How does this affect the GRN[7]? Are results still reliable in these cases? Can we quantify uncertainty of cell-specific graphs and relate that to graph properties or metadata?

3. Added value of ATACseq data: ATACseq data is one of the types of data integrated in this model. It offers a complementary view of the biological state, when combined with more “traditional” RNAseq data. How much does the GRN change if made without the ATAC data?

3b. For some samples in our dataset these modalities as paired, which means that the measurements are from the exact same cell. For others they are artificially aligned with a method called Harmony[8]. What is the effect of these different approaches[9]?

4. Without HVG selection: The paper relies on first selecting only the 1,000 most highly variable genes (HVGs)[10]. While these are often a good summary of the data, they also ignore a lot of details. Can we build a model that uses HVG selection only where necessary for computational reasons, but otherwise model the relationship between all ~36,000 genes[11]?

5.  Spatial variability: We also have spatial gene expression data available, which we can use to map the single-cell data back to their physical locations in the brain[12]. Can we relate the spatial location to changes in the network structure?

6. (extra) GRN prior: Other methods, like digNET[13], construct cell-specific GRNs by starting from an initial global GRN and then refining it. Can we incorporate this into scReNI? Possibly in a Bayesian fashion[14]?

## References
[1] Alzheimer’s Association. (2024). What is Alzheimer’s disease? https://www.alz.org/alzheimers-dementia/what-is-alzheimers
[2] National Institute on Aging. (2023). What happens to the brain in Alzheimer’s disease? National Institutes of Health. https://www.nia.nih.gov/health/what-happens-brain-alzheimers-disease
[3] Ouyang, W., et al. (2024). Gene regulatory networks in physiology and disease. Nature Reviews Nephrology, 20, 741–760. https://doi.org/10.1038/s41581-024-00849-7
[4] Xu, X., Liang, Y., & Tang, M. (2025). ScReNI: Single-cell regulatory network inference through integrating scRNA-seq and scATAC-seq data. Genomics, Proteomics & Bioinformatics. https://doi.org/10.1093/gpbjnl/qzaf060
[5] Guo, Z., He, Y., & Han, L. (2021). Identifying regulatory network modules associated with breast cancer stages. Frontiers in Genetics, 12, 717557. https://doi.org/10.3389/fgene.2021.717557
[6] Zanin, E., et al. (2017). Differential network biology. Cellular and Molecular Life Sciences, 74, 3177–3201. https://doi.org/10.1007/s00018-017-2679-6
[7] Bragina, A. (Ed.). (2014). Analysis of Biological Networks [PDF]. retrieved from institutional repository. (foundational textbook on biological networks analysis) https://www.researchgate.net/profile/Anastasia-Bragina/post/Can_anybody_suggest_references_for_the_analysis_of_microbial_co-occurrance_networks/attachment/59d63ddbc49f478072ea8bb2/AS%3A273764120498181%401442281860593/download/BOOK_Analysis+of+biological+networks.pdf
[8] Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. Nature Methods, 16(12), 1289–1296. https://doi.org/10.1038/s41592-019-0619-0
[9] Lee, M. Y. Y., Kaestner, K. H., & Li, M. (2023). Benchmarking algorithms for joint integration of unpaired and paired single-cell RNA-seq and ATAC-seq data. Genome Biology, 24(1), 244. https://doi.org/10.1186/s13059-023-03073-x
[10] Theis, F. J., et al. (2019). Best practices for single-cell RNA-seq analysis: a tutorial. Molecular Systems Biology, 15(6), e88746. https://doi.org/10.15252/msb.20188746
[11] Wagner, F., Yan, Y., & Yanai, I. (2018). K-nearest neighbor smoothing for high-throughput single-cell RNA-Seq data [Preprint]. bioRxiv. https://doi.org/10.1101/217737
[12] Abdelaal, T., et al. (2020). spaGE: spatial gene enhancement using transfer learning for spatial transcriptomics. Nucleic Acids Research, 48(18), e107. https://doi.org/10.1093/nar/gkaa771
[13] Fortelny, N., et al. (2025). digNET: dissecting gene regulatory networks from biological systems. Genome Research, 35(2), 340–350.https://genome.cshlp.org/content/35/2/340.full
[14] MERLIN+Prior: Bayesian gene regulatory network inference with prior knowledge. Nucleic Acids Research, 45(4), e21. https://doi.org/10.1093/nar/gkw1127