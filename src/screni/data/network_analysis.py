from 


def network_analysis(regulatory_relationships, kmeans_result, TFFDR1: float = 0.1, TFFDR2: float = 0.5, ModuleFDR: float = 0.05):

    if ("Correlation" not in regulatory_relationships.columns):
        raise ValueError("regulatory_relationships should contain Correlation column")

    if ("TF" not in regulatory_relationships.columns):
        raise ValueError("regulatory_relationships should contain TF column")

    if ("Target" not in regulatory_relationships.columns):
        raise ValueError("regulatory_relationships should contain Target column")

    if ("KmeansGroup" not in kmeans_result.columns):
        raise ValueError("kmeans_result should contain KmeansGroup column")

    valid_genes = set(kmeans_result.index)

    for col in ["TF", "Target"]:
        missing = set(regulatory_relationships[col]) - valid_genes
        if missing:
            raise ValueError(f"The following {col}s are missing from kmeans_result: {list(missing)[:5]}...")
            
    TFs_list = get_Enriched_TFs(regulatory_relationships, Kmeans_result,
        TFFdrThr1 = TFFDR1)

    TFs_list = get_regulation_of_TFs_to_modules(TFs_list, TFFDR2)
    TFs_list = get_partial_regulations(TFs_list)
    TFs_list = merge_Module_Regulations(TFs_list, Kmeans_result, ModuleThr1 = ModuleFDR)
    return(TFs_list)
# }