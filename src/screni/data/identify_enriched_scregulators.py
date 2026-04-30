from network_analysis import network_analysis

def identify_enriched_scregulators(regulatory_network, kmeans_clustering_ENS, TFFDR1: float = 0.1, TFFDR2: float = 0.1):
    """
    regulatory_network :  The GRN
    kmeans_clustering_ENS : The grouping of cells, showing which genes belong to which family
    TFFDR1 : False Discovery Rate (FDR) threshold used to identify enriched transcription factors
    TFFDR2 : False Discovery Rate (FDR) threshold to refine the strength of connection between TFs and gene modules.
    
    High-level API wrapper, made for readability. Calls the method `network_analysis`

    """
    TFs_list = network_analysis(regulatory_network, kmeans_clustering_ENS, TFFDR1 = TFFDR1, TFFDR2 = TFFDR2)
    return TFs_list