"""Data loading, integration, and network utilities for ScReNI."""

from screni.data.clustering import (
    NetworkDegreeResult,
    calculate_scnetwork_degree,
    clustering_kmeans,
    enrich_module,
)
from screni.data.combine import ScReniNetworks, combine_wscreni_networks
from screni.data.evaluation import (
    calculate_network_precision_recall,
    calculate_network_precision_recall_top,
    load_chip_atlas,
    summary_se,
)
from screni.data.inference import (
    GenePeakOverlapLabs,
    infer_csn_networks,
    infer_kscreni_networks,
    infer_lioness_networks,
    infer_wscreni_networks,
)
from screni.data.precision_recall import (
    calculate_precision_recall,
    deal_gene_information,
)
from screni.data.regulator_enrichment import (
    identify_enriched_scregulators,
    network_analysis,
)
from screni.data.differential import (
    DonorPseudobulks,
    EdgeTestResult,
    benjamini_hochberg,
    differential_edges,
    load_run_dir,
    ols_with_covariates,
    pseudobulk_per_donor,
    wilcoxon_unadjusted,
)
from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
    subsample_cells_per_donor,
)

__all__ = [
    # inference
    "infer_kscreni_networks",
    "infer_wscreni_networks",
    "GenePeakOverlapLabs",
    "infer_csn_networks",
    "infer_lioness_networks",
    # combine
    "ScReniNetworks",
    "combine_wscreni_networks",
    # precision_recall (Leo)
    "deal_gene_information",
    "calculate_precision_recall",
    # evaluation (Ivo)
    "summary_se",
    "calculate_network_precision_recall",
    "calculate_network_precision_recall_top",
    "load_chip_atlas",
    # clustering (Minhea)
    "NetworkDegreeResult",
    "calculate_scnetwork_degree",
    "clustering_kmeans",
    "enrich_module",
    # regulator enrichment (Duco)
    "network_analysis",
    "identify_enriched_scregulators",
    # differential analysis (Ivo, SQ1)
    "DonorPseudobulks",
    "EdgeTestResult",
    "benjamini_hochberg",
    "differential_edges",
    "load_run_dir",
    "ols_with_covariates",
    "pseudobulk_per_donor",
    "wilcoxon_unadjusted",
    # SeaAD SQ1 helpers
    "add_condition_column",
    "add_copathology_columns",
    "select_eligible_donors",
    "subsample_cells_per_donor",
]
