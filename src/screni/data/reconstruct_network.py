import numpy as np
import pandas as pd
import re
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

## Faithfull python port of https://github.com/Xuxl2020/ScReNI/blob/master/R/reconstruct_network_part.R


def get_Enriched_TFs(GeneCor1: pd.DataFrame, Kmeans_result: pd.DataFrame, TFFdrThr1: float = 2) -> dict:
    """
    Identify transcription factors (TFs) enriched in gene co-expression modules
    using hypergeometric testing.

    Parameters
    ----------
    GeneCor1 : pd.DataFrame
        Edge table with columns: 'TF', 'Correlation', 'Target' (plus others).
    Kmeans_result : pd.DataFrame
        Gene-level DataFrame indexed by gene name with a 'KmeansGroup' column.
    TFFdrThr1 : float
        -log10(FDR) threshold for enrichment (default: 2, i.e. FDR < 0.01).

    Returns
    -------
    dict with keys:
        'Cor_TFs'                    - all TFs with per-module hypergeometric stats
        'Cor_EnTFs'                  - enriched TFs only
        'FOSF_RegMTF_Cor_EnTFs'      - edges whose regulator is an enriched TF
        'FOSF_RegMTF_Cor_EnTFsTarg'  - edges where both TF and target are enriched TFs
        'FOSF_RegMTF_Cor_EnTFsTargM' - above, restricted to within-module edges
    """
    GeneCor1 = GeneCor1.copy()
    GeneCor1["TF"] = GeneCor1["TF"].astype("category")

    GeneCorP1 = GeneCor1[GeneCor1["Correlation"] > 0].copy()
    GeneCorN1 = GeneCor1[GeneCor1["Correlation"] < 0].copy()
    Module1 = Kmeans_result.copy()

    TF0 = GeneCor1["TF"].value_counts().reindex(GeneCor1["TF"].cat.categories, fill_value=0)
    TF01 = pd.Series(0, index=TF0.index)

    TFP01 = GeneCorP1["TF"].value_counts()
    TFN01 = GeneCorN1["TF"].value_counts()

    TFP1 = TF01.copy()
    TFP1[TFP01.index] = TFP01.values

    TFN1 = TF01.copy()
    TFN1[TFN01.index] = TFN01.values

    uGroup1 = sorted(Module1["KmeansGroup"].unique())
    pTF4 = None

    for i, group in enumerate(uGroup1):
        Module2 = Module1.index[Module1["KmeansGroup"] == group].tolist()

        for j in range(2):
            if j == 0:
                GeneCorPN = GeneCorP1
                TF1 = TFP1
                GeneCor2 = GeneCorP1[GeneCorP1["Target"].isin(Module2)]
            else:
                GeneCorPN = GeneCorN1
                TF1 = TFN1
                GeneCor2 = GeneCorN1[GeneCorN1["Target"].isin(Module2)]

            TF02 = GeneCor2["TF"].value_counts()
            TF2 = TF01.copy()
            TF2[TF02.index] = TF02.values

            TF3 = pd.DataFrame({
                "x": TF2,
                "M": TF1,
                "N": len(GeneCorPN) - TF1,
                "n": len(GeneCor2),
            })

            def hypergeom_test(row):
                x, M, N, n = int(row["x"]), int(row["M"]), int(row["N"]), int(row["n"])
                if x == 0 or (x < 5 and x < 0.02 * n):
                    return 1.0
                return hypergeom.sf(x - 1, M + N, M, n)

            pTF3 = TF3.apply(hypergeom_test, axis=1)
            _, pTF31, _, _ = multipletests(pTF3.values, method="fdr_bh")
            pTF31 = pd.Series(pTF31, index=pTF3.index)

            num_str = TF3.apply(lambda r: ";".join(str(int(v)) for v in r), axis=1)

            direction = "P" if j == 0 else "N"
            block = pd.DataFrame({
                f"{direction}num{group}": num_str,
                f"{direction}p{group}":   pTF3,
                f"{direction}fdr{group}": pTF31,
            })

            pTF4 = block if pTF4 is None else pd.concat([pTF4, block], axis=1)

    fdr_cols = [c for c in pTF4.columns if "fdr" in c.lower()]

    def summarize_tf(row):
        vals = pd.to_numeric(row[fdr_cols], errors="coerce").fillna(1.0)
        nlog = -np.log10(vals.clip(lower=1e-300))

        pos_nlog = nlog.iloc[0::2].values
        neg_nlog = nlog.iloc[1::2].values

        sig_act = [str(uGroup1[k]) for k, v in enumerate(pos_nlog) if v > TFFdrThr1]
        sig_rep = [str(uGroup1[k]) for k, v in enumerate(neg_nlog) if v > TFFdrThr1]

        x212 = ";".join(sig_act) if sig_act else "NA"
        x222 = ";".join(sig_rep) if sig_rep else "NA"

        max_val = nlog.max()
        max_idx = nlog.argmax()
        direction = "P" if max_idx % 2 == 0 else "N"
        module_num = uGroup1[max_idx // 2]
        ix3 = f"{direction}{module_num}"

        return pd.Series(
            [max_val, ix3, x212, x222],
            index=["TFMinNlogfdr", "TFMinGroup", "SigActModules", "SigRepModules"],
        )

    pTF4Min = pTF4.apply(summarize_tf, axis=1)
    pTF4Min["TFMinNlogfdr"] = pd.to_numeric(pTF4Min["TFMinNlogfdr"])

    pTF4Mod = Module1.reindex(pTF4.index)
    pTF42 = pd.concat([pTF4Mod, pTF4Min, pTF4], axis=1)

    Ind42 = pTF42.index[pTF42["TFMinNlogfdr"] > TFFdrThr1]
    EnrichTF1 = pTF42.loc[Ind42]
    print(f"Total TFs: {len(pTF42)}")
    print(f"Enriched TFs: {len(EnrichTF1)}")

    tf_str = GeneCor1["TF"].astype(str)
    GeneCorTFfdr = pTF4Min.reindex(tf_str.values)
    GeneCorTFfdr.index = GeneCor1.index

    Regulation = GeneCor1["Correlation"].apply(
        lambda x: "Negative" if float(x) < 0 else "Positive"
    )

    tf_col = GeneCor1[["TF"]]
    target_cols = GeneCor1.loc[:, "Target":]
    GeneCor3 = pd.concat(
        [tf_col, GeneCorTFfdr, target_cols, Regulation.rename("Regulation")],
        axis=1,
    )

    EnTFReg1  = GeneCor3[GeneCor3["TF"].isin(EnrichTF1.index)]
    EnTFTarg1 = GeneCor3[
        GeneCor3["TF"].isin(EnrichTF1.index) &
        GeneCor3["Target"].isin(EnrichTF1.index)
    ]

    EnTFGroup1 = sorted(
        set(EnTFTarg1["TFGroup"].unique()) | set(EnTFTarg1["TargetGroup"].unique())
    )
    EnTFTarg2 = pd.concat(
        [
            EnTFTarg1[
                (EnTFTarg1["TFGroup"] == g) & (EnTFTarg1["TargetGroup"] == g)
            ]
            for g in EnTFGroup1
        ],
        ignore_index=True,
    )

    return {
        "Cor_TFs":                    pTF42,
        "Cor_EnTFs":                  EnrichTF1,
        "FOSF_RegMTF_Cor_EnTFs":      EnTFReg1,
        "FOSF_RegMTF_Cor_EnTFsTarg":  EnTFTarg1,
        "FOSF_RegMTF_Cor_EnTFsTargM": EnTFTarg2,
    }


def get_regulation_of_TFs_to_modules(TFs_list: dict, Thr: float = 2) -> dict:
    """
    Build a long-form TF × module regulation table from enriched TF results.

    Parameters
    ----------
    TFs_list : dict
        Output from get_Enriched_TFs (or a downstream step).
    Thr : float
        -log10(FDR) threshold (default: 2).

    Returns
    -------
    Updated TFs_list with added keys 'TF_list' and 'TF_module_regulation'.
    """
    con1 = TFs_list["Cor_EnTFs"].copy()
    fdr_pattern = re.compile(r"^[PN]fdr\d+$")
    fdr_cols = [(j, col) for j, col in enumerate(con1.columns) if fdr_pattern.match(col)]

    rows = []
    tf_list = []

    for i, row in con1.iterrows():
        for j, col_name in fdr_cols:
            val = row.iloc[j]
            fdr1 = np.inf if val == 0 else -np.log10(float(val))
            if fdr1 > Thr:
                regulation = "Positive" if col_name.startswith("P") else "Negative"
                tg = re.search(r"\d+", col_name).group()
                rows.append({
                    "TF":           i,
                    "TFSymbol":     row.iloc[1],
                    "TFGroup":      row.iloc[0],
                    "TargetModule": f"Group{tg}",
                    "TargetGroup":  tg,
                    "Regulation":   regulation,
                    "Nlogfdr":      fdr1,
                })
                tf_list.append(row.iloc[1])

    TF_module_regulation = pd.DataFrame(rows)
    if not TF_module_regulation.empty:
        TF_module_regulation["Nlogfdr"] = pd.to_numeric(TF_module_regulation["Nlogfdr"])

    TF_list_unique = list(dict.fromkeys(tf_list))

    TFs_list = dict(TFs_list)
    TFs_list["TF_list"] = TF_list_unique
    TFs_list["TF_module_regulation"] = TF_module_regulation
    return TFs_list


def get_partial_regulations(TFs_list: dict) -> dict:
    """
    Subset the TF regulatory network to edges where both regulator and target
    are in the enriched TF list.

    Mirrors R's column-position checks:
        con1[i,][2]  -> TFSymbol  (iloc[:,1])
        con1[i,][9]  -> TargetSymbol (iloc[:,8])

    Parameters
    ----------
    TFs_list : dict
        Must contain 'FOSF_RegMTF_Cor_EnTFs' and 'TF_list'.

    Returns
    -------
    Updated TFs_list with added key 'TF_network'.
    """
    con1 = TFs_list["FOSF_RegMTF_Cor_EnTFs"].copy()
    hash2 = set(TFs_list["TF_list"])

    con1["TFSymbol"]     = con1["TFSymbol"].astype(str)
    con1["TargetSymbol"] = con1["TargetSymbol"].astype(str)

    mask = con1["TFSymbol"].isin(hash2) & con1["TargetSymbol"].isin(hash2)
    col1 = con1[mask].copy()

    TFs_list = dict(TFs_list)
    TFs_list["TF_network"] = col1
    return TFs_list


def merge_Module_Regulations(
    TFs_list: dict,
    Kmeans_result: pd.DataFrame,
    ModuleThr1: float = 0.05,
) -> dict:
    """
    Test for enrichment of TF→module regulatory edges between every pair of
    modules using the hypergeometric distribution, then FDR-correct.

    Parameters
    ----------
    TFs_list : dict
        Must contain 'FOSF_RegMTF_Cor_EnTFsTarg'.
    Kmeans_result : pd.DataFrame
        DataFrame with 'KmeansGroup' column.
    ModuleThr1 : float
        FDR threshold (default: 0.05).

    Returns
    -------
    Updated TFs_list with added key 'intramodular_network'.
    """
    TF1 = Kmeans_result.copy()
    Regulation1 = TFs_list["FOSF_RegMTF_Cor_EnTFsTarg"].copy()
    Regulation1["Correlation"] = pd.to_numeric(Regulation1["Correlation"])
    Module1 = sorted(TF1["KmeansGroup"].unique())

    RegulationP = Regulation1[Regulation1["Regulation"] == "Positive"]
    RegulationN = Regulation1[Regulation1["Regulation"] == "Negative"]

    records = []

    for i, mod_i in enumerate(Module1):
        Reg_i  = Regulation1[Regulation1["TFGroup"] == mod_i]
        Reg_iP = Reg_i[Reg_i["Regulation"] == "Positive"]
        Reg_iN = Reg_i[Reg_i["Regulation"] == "Negative"]
        Target_iP = RegulationP[RegulationP["TargetGroup"] == mod_i]
        Target_iN = RegulationN[RegulationN["TargetGroup"] == mod_i]

        for j in range(i, len(Module1)):
            mod_j = Module1[j]
            Reg_j  = Regulation1[Regulation1["TFGroup"] == mod_j]
            Reg_jP = Reg_j[Reg_j["Regulation"] == "Positive"]
            Reg_jN = Reg_j[Reg_j["Regulation"] == "Negative"]
            Target_jP = RegulationP[RegulationP["TargetGroup"] == mod_j]
            Target_jN = RegulationN[RegulationN["TargetGroup"] == mod_j]

            def make_record(tf_grp, tgt_grp, reg_dir, edges, target_in_tgt, all_dir, tf_edges):
                return {
                    "TFGroup":     tf_grp,
                    "TargetGroup": tgt_grp,
                    "Regulation":  reg_dir,
                    "Correlation": edges["Correlation"].mean() if len(edges) else float("nan"),
                    "x": len(edges),
                    "M": len(target_in_tgt),
                    "N": len(all_dir) - len(target_in_tgt),
                    "n": len(tf_edges),
                }

            records.append(make_record(
                mod_i, mod_j, "Positive",
                Reg_iP[Reg_iP["TargetGroup"] == mod_j],
                Target_jP, RegulationP, Reg_iP,
            ))
            records.append(make_record(
                mod_i, mod_j, "Negative",
                Reg_iN[Reg_iN["TargetGroup"] == mod_j],
                Target_jN, RegulationN, Reg_iN,
            ))

            if i != j:
                records.append(make_record(
                    mod_j, mod_i, "Positive",
                    Reg_jP[Reg_jP["TargetGroup"] == mod_i],
                    Target_iP, RegulationP, Reg_jP,
                ))
                records.append(make_record(
                    mod_j, mod_i, "Negative",
                    Reg_jN[Reg_jN["TargetGroup"] == mod_i],
                    Target_iN, RegulationN, Reg_jN,
                ))

    RegulationNum1 = pd.DataFrame(records).dropna(subset=["Correlation"])

    def hyper_p(row):
        x, M, N, n = int(row["x"]), int(row["M"]), int(row["N"]), int(row["n"])
        if x < 4:
            return 1.0
        return hypergeom.sf(x - 1, M + N, M, n)

    RegulationP1 = RegulationNum1.apply(hyper_p, axis=1).values
    _, fdr_adj, _, _ = multipletests(RegulationP1, method="fdr_bh")
    RegulationP2 = -np.log10(fdr_adj)

    num_str = RegulationNum1[["x", "M", "N", "n"]].apply(
        lambda r: ";".join(str(int(v)) for v in r), axis=1
    )

    RegulationP3 = RegulationNum1[["TFGroup", "TargetGroup", "Regulation", "Correlation"]].copy()
    RegulationP3["NumberRegulation"] = num_str.values
    RegulationP3["Pvalue"]   = RegulationP1
    RegulationP3["NlogFdr"]  = RegulationP2

    RegulationP4 = RegulationP3[RegulationP3["NlogFdr"] > -np.log10(ModuleThr1)]
    RegulationP4 = RegulationP4.sort_values(["TFGroup", "TargetGroup"]).reset_index(drop=True)
    print(f"Significant regulations: {len(RegulationP4)}")

    TFs_list = dict(TFs_list)
    TFs_list["intramodular_network"] = RegulationP4
    return TFs_list
