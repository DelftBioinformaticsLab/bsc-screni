"""Pilot/full driver for wScReNI inference on SEA-AD paired multiome.

Glues together the existing Phase 2 (feature selection + KNN), Phase 3
(gene-peak triplets), and wScReNI inference functions, parameterised so
the same script handles:

  * the pilot run (small donor list, 30 cells/donor)
  * the full SQ1 run per cell type (all eligible donors, 50 cells/donor)
  * sensitivity runs (different ``--cells-per-donor``, different donor lists)

Outputs are written under ``--output-dir`` keyed by cell type so that the
two cell-type runs (Microglia-PVM, L2/3 IT) don't collide:

    <output-dir>/
        <cell_type_safe>/
            seaad_<ct>_rna_sub.h5ad
            seaad_<ct>_atac_sub.h5ad
            seaad_<ct>_knn_indices.npy
            seaad_<ct>_triplets.csv
            seaad_<ct>_gene_labels.csv
            seaad_<ct>_peak_overlap_matrix.npz
            seaad_<ct>_donor_metadata.csv     (one row per included donor)
            wScReNI/                           (per-cell weight matrices)
                <i>.<cell_name>.network.txt

After this script completes, ``differential.py`` consumes the wScReNI
output dir + the donor metadata to produce the per-cell-type ranked
TF->target table.

Example
-------
Pilot (5 donors, 30 cells each, Microglia)::

    python scripts/run_seaad_inference.py \\
        --cell-type "Microglia-PVM" \\
        --donors H21.33.003,H20.33.002,H21.33.019,H20.33.004,H20.33.008 \\
        --cells-per-donor 30 \\
        --output-dir output/seaad_pilot \\
        --pilot

Full SQ1 run (Microglia)::

    python scripts/run_seaad_inference.py \\
        --cell-type "Microglia-PVM" \\
        --cells-per-donor 50 \\
        --output-dir output/seaad_sq1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd

from screni.data.feature_selection import prepare_subsample
from screni.data.gene_peak_relations import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_DOWNSTREAM_BP,
    DEFAULT_MOTIF_PVALUE,
    DEFAULT_UPSTREAM_BP,
    load_transfac_motifs,
    run_phase3,
)
from screni.data.inference import GenePeakOverlapLabs, infer_wscreni_networks
from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
    subsample_cells_per_donor,
)
from screni.data.utils import load_gene_annotations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run wScReNI on a SEA-AD paired-multiome cell-type slice."
    )
    p.add_argument("--cell-type", required=True, help='e.g. "Microglia-PVM"')
    p.add_argument(
        "--donors",
        default=None,
        help=(
            "Comma-separated donor IDs to include.  If omitted, all donors "
            "with >=--min-cells-per-donor cells in --cell-type and a "
            "control/ad condition are used."
        ),
    )
    p.add_argument("--cells-per-donor", type=int, default=50)
    p.add_argument("--min-cells-per-donor", type=int, default=50)

    p.add_argument(
        "--rna-h5ad",
        default="data/processed/seaad/seaad_paired_rna.h5ad",
    )
    p.add_argument(
        "--atac-h5ad",
        default="data/processed/seaad/seaad_paired_atac.h5ad",
    )
    p.add_argument(
        "--integrated-h5mu",
        default="data/processed/seaad/seaad_paired_integrated.h5mu",
    )
    p.add_argument(
        "--gtf",
        default="data/paper/reference/gtf_regions.GRCh38.txt",
        help="Pre-parsed gene annotations TSV (columns: chr, start, end, gene_id, gene_name, strand).",
    )
    p.add_argument(
        "--motif-db",
        default="data/paper/reference/Tranfac201803_Hs_MotifTFsFinal",
    )
    p.add_argument(
        "--motif-pwm",
        default="data/paper/reference/all_motif_pwm.rds",
    )
    p.add_argument(
        "--genome-fasta",
        default="data/reference/hg38.fa",
    )
    p.add_argument("--gene-name-type", default="symbol", choices=["symbol", "id"])
    p.add_argument("--n-genes", type=int, default=500, help="HVG count for Phase 2.")
    p.add_argument("--n-peaks", type=int, default=10000, help="HVP count for Phase 2.")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    p.add_argument("--motif-pvalue", type=float, default=DEFAULT_MOTIF_PVALUE)

    p.add_argument("--n-trees", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help="Stop after Phase 3, useful for verifying triplet generation.",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: time the first cell, log a runtime projection.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_name(s: str) -> str:
    """Filesystem-safe slug (e.g. 'L2/3 IT' -> 'L2_3_IT')."""
    return s.replace("/", "_").replace(" ", "_").replace("-", "_")


def _load_seaad_paired(
    rna_path: Path, atac_path: Path, cell_type: str
) -> tuple[ad.AnnData, ad.AnnData]:
    """Load paired SEA-AD multiome filtered to one cell type and to cells
    with a defined ``condition`` (control / ad)."""
    logger.info(f"Loading paired RNA from {rna_path} ...")
    rna = ad.read_h5ad(rna_path)
    logger.info(f"  RNA: {rna.shape}")

    logger.info(f"Loading paired ATAC from {atac_path} ...")
    atac = ad.read_h5ad(atac_path)
    logger.info(f"  ATAC: {atac.shape}")

    # cell_type column from the loader
    if "cell_type" not in rna.obs.columns and "Subclass" in rna.obs.columns:
        rna.obs["cell_type"] = rna.obs["Subclass"].astype(str)
    if "cell_type" not in atac.obs.columns and "Subclass" in atac.obs.columns:
        atac.obs["cell_type"] = atac.obs["Subclass"].astype(str)

    # Add condition + co-pathology
    add_condition_column(rna)
    add_condition_column(atac)
    add_copathology_columns(rna)
    add_copathology_columns(atac)

    # Filter to chosen cell type + cells with a defined condition
    rna_mask = (rna.obs["cell_type"] == cell_type) & rna.obs["condition"].notna()
    atac_mask = (atac.obs["cell_type"] == cell_type) & atac.obs["condition"].notna()
    rna_ct = rna[rna_mask].copy()
    atac_ct = atac[atac_mask].copy()
    logger.info(
        f"  After {cell_type} + condition filter: "
        f"RNA={rna_ct.shape}, ATAC={atac_ct.shape}"
    )
    return rna_ct, atac_ct


def _align_paired(
    rna: ad.AnnData, atac: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    """Restrict both AnnDatas to the cells present in both, in identical order."""
    shared = sorted(set(rna.obs_names) & set(atac.obs_names))
    if len(shared) == 0:
        raise RuntimeError("No shared cell barcodes between RNA and ATAC.")
    if len(shared) < min(rna.n_obs, atac.n_obs):
        logger.warning(
            f"  Pairing reduced cells: RNA={rna.n_obs}, ATAC={atac.n_obs}, "
            f"shared={len(shared)}"
        )
    return rna[shared].copy(), atac[shared].copy()


def _extract_wnn_embedding(
    integrated_h5mu: Path, target_cells: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Pull the WNN (or PCA fallback) embedding for the target cells.

    Returns
    -------
    embedding : (n_target, n_dims) float
    cell_names : list[str], same order as embedding rows
    """
    logger.info(f"Loading WNN embedding from {integrated_h5mu} ...")
    mdata = mu.read(str(integrated_h5mu))
    rna_mod = mdata.mod.get("rna")
    if rna_mod is None:
        raise RuntimeError("integrated h5mu has no 'rna' modality.")

    emb = None
    used = None
    for key in ("X_wnn", "X_pca"):
        if key in rna_mod.obsm:
            emb = np.asarray(rna_mod.obsm[key])
            used = key
            break
    if emb is None:
        raise RuntimeError(
            f"No X_wnn or X_pca in integrated rna obsm. "
            f"Have: {list(rna_mod.obsm.keys())}"
        )
    full_names = list(rna_mod.obs_names)
    logger.info(f"  Using {used}: {emb.shape}")

    # Subset to target cells, preserving target order
    name_to_row = {n: i for i, n in enumerate(full_names)}
    rows = [name_to_row[c] for c in target_cells if c in name_to_row]
    if len(rows) != len(target_cells):
        missing = set(target_cells) - set(full_names)
        raise RuntimeError(
            f"{len(missing)} target cells not in integrated obsm "
            f"(e.g. {sorted(missing)[:3]})"
        )
    return emb[rows], list(target_cells)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_root = Path(args.output_dir)
    ct_dir = out_root / _safe_name(args.cell_type)
    ct_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"seaad_{_safe_name(args.cell_type)}"

    logger.info("=" * 70)
    logger.info(f"SEA-AD wScReNI driver | cell_type={args.cell_type}")
    logger.info(f"  output: {ct_dir}")
    logger.info(f"  pilot:  {args.pilot}")
    logger.info("=" * 70)

    # ---- Step 1: load + condition ----
    rna_ct, atac_ct = _load_seaad_paired(
        Path(args.rna_h5ad), Path(args.atac_h5ad), args.cell_type
    )

    # ---- Step 2: pick eligible donors ----
    if args.donors:
        chosen = [d.strip() for d in args.donors.split(",") if d.strip()]
        donor_table = (
            rna_ct.obs.loc[rna_ct.obs["Donor ID"].isin(chosen)]
            .groupby("Donor ID", observed=True)
            .first()
            .reset_index()
            .rename(columns={"Donor ID": "donor_id"})
        )[["donor_id", "condition"]]
        donor_table["n_cells"] = (
            rna_ct.obs.groupby("Donor ID", observed=True).size().reindex(chosen).values
        )
        logger.info(f"User-supplied donor list: {len(chosen)} donors")
    else:
        donor_table = select_eligible_donors(
            rna_ct,
            cell_type=args.cell_type,
            min_cells_per_donor=args.min_cells_per_donor,
        )

    if donor_table.empty:
        logger.error("No donors meet the eligibility criteria; aborting.")
        return 1

    donor_table.to_csv(ct_dir / f"{prefix}_donor_metadata.csv", index=False)
    chosen_donors = donor_table["donor_id"].tolist()

    # ---- Step 3: subsample cells per donor ----
    rna_ct = rna_ct[rna_ct.obs["Donor ID"].isin(chosen_donors)].copy()
    atac_ct = atac_ct[atac_ct.obs["Donor ID"].isin(chosen_donors)].copy()
    rna_sub = subsample_cells_per_donor(
        rna_ct,
        n_per_donor=args.cells_per_donor,
        seed=args.seed,
    )
    atac_sub, _ = _align_paired(atac_ct, rna_sub)
    rna_sub, atac_sub = _align_paired(rna_sub, atac_sub)
    logger.info(f"  After paired alignment: {rna_sub.n_obs} cells")

    # ---- Step 4: WNN embedding + Phase 2 (feature selection + KNN) ----
    target_cells = list(rna_sub.obs_names)
    embedding, embedding_names = _extract_wnn_embedding(
        Path(args.integrated_h5mu), target_cells
    )

    phase2 = prepare_subsample(
        rna=rna_sub,
        atac=atac_sub,
        n_per_type=10**9,  # already subsampled by donor; let phase2 keep them all
        n_genes=args.n_genes,
        n_peaks=args.n_peaks,
        seed=args.seed,
        embedding=embedding,
        embedding_cell_names=embedding_names,
        knn_k=args.knn_k,
    )
    rna_p2 = phase2["rna"]
    atac_p2 = phase2["atac"]
    knn_indices = phase2.get("knn_indices")
    if knn_indices is None:
        raise RuntimeError("Phase 2 did not produce KNN indices.")

    rna_p2.write_h5ad(ct_dir / f"{prefix}_rna_sub.h5ad")
    atac_p2.write_h5ad(ct_dir / f"{prefix}_atac_sub.h5ad")
    np.save(ct_dir / f"{prefix}_knn_indices.npy", knn_indices)
    logger.info(
        f"  Phase 2 done: {rna_p2.shape} RNA, {atac_p2.shape} ATAC, "
        f"KNN={knn_indices.shape}"
    )

    # ---- Step 5: Phase 3 (gene-peak triplets) ----
    gene_annotations = load_gene_annotations(args.gtf)
    pwm_dict, motif_db = load_transfac_motifs(args.motif_pwm, args.motif_db)
    phase3 = run_phase3(
        rna_adata=rna_p2,
        atac_adata=atac_p2,
        gene_annotations=gene_annotations,
        genome_fasta=args.genome_fasta,
        pwm_dict=pwm_dict,
        motif_db=motif_db,
        upstream_bp=DEFAULT_UPSTREAM_BP,
        downstream_bp=DEFAULT_DOWNSTREAM_BP,
        corr_threshold=args.corr_threshold,
        motif_pvalue=args.motif_pvalue,
        gene_name_type=args.gene_name_type,
        output_dir=ct_dir,
        prefix=prefix,
    )
    triplets = phase3["triplets"]
    peak_matrix = phase3["peak_matrix"]
    peak_info = phase3["peak_info"]
    logger.info(f"  Phase 3 done: {len(triplets)} TF->target triplets")

    if args.skip_inference:
        logger.info("--skip-inference set; stopping after Phase 3.")
        return 0

    # ---- Step 6: assemble GenePeakOverlapLabs and run wScReNI ----
    # GenePeakOverlapLabs.from_dataframe expects columns
    # ['gene.name', 'peak.name', 'TF']; triplets has ['TF', 'peak', 'target_gene', ...].
    labs_df = triplets.rename(columns={
        "target_gene": "gene.name",
        "peak": "peak.name",
    })[["gene.name", "peak.name", "TF"]]
    labs = GenePeakOverlapLabs.from_dataframe(labs_df)

    # Reorient: wScReNI expects expr (cells x genes), peak_mat (cells x peaks)
    expr = rna_p2  # already cells x genes
    peak_mat_ad = ad.AnnData(
        X=peak_matrix,
        obs=atac_p2.obs.copy(),
        var=pd.DataFrame(index=peak_info["peak"].tolist()),
    )

    t0 = time.perf_counter()
    networks = infer_wscreni_networks(
        expr=expr,
        peak_mat=peak_mat_ad,
        labs=labs,
        nearest_neighbors_idx=knn_indices,
        network_path=ct_dir,
        data_name=f"seaad_{_safe_name(args.cell_type)}",
        cell_index=None,
        n_jobs=args.n_jobs,
        max_cells_per_batch=10,
        n_trees=args.n_trees,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        f"  wScReNI done: {len(networks)} cells in {elapsed/60:.1f} min "
        f"({elapsed/max(len(networks),1):.1f}s/cell)"
    )

    # Persist a small run-summary JSON so differential.py can find everything
    summary = {
        "cell_type": args.cell_type,
        "n_donors": len(chosen_donors),
        "n_cells": int(rna_p2.n_obs),
        "n_genes": int(rna_p2.n_vars),
        "n_peaks": int(atac_p2.n_vars),
        "n_triplets": int(len(triplets)),
        "elapsed_sec": elapsed,
        "args": vars(args),
        "donor_metadata_path": str(ct_dir / f"{prefix}_donor_metadata.csv"),
        "networks_dir": str(ct_dir / "wScReNI"),
    }
    (ct_dir / f"{prefix}_run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    if args.pilot:
        logger.info("Pilot projection (assumes linear scaling):")
        for n_target in (500, 1000, 1500, 2000):
            proj = elapsed * n_target / max(len(networks), 1) / 3600
            logger.info(f"  {n_target} cells -> {proj:.1f} h wall-clock")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
