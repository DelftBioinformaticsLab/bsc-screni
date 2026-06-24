#!/usr/bin/env python3
"""
run_seaad_hvg_selection_microglia_v2.py
========================================
Fixed version of run_seaad_hvg_selection_microglia.py with three corrections:

  FIX 1 — Raw counts source
    v1 fell back to seaad_paired_rna.h5ad .X, which holds log-normalized
    values (max ~6), causing VST to estimate wrong variances.  This version
    tries raw count layers in seaad_paired_rna.h5ad first, then falls back
    to seaad_paired_integrated.h5mu (confirmed source of raw counts).

  FIX 2 — 2000 HVGs instead of 500
    More genes → higher probability that microglia-variable TFs survive
    selection, and more gene-peak proximity pairs for Phase 3 correlation.

  FIX 3 — All 3,129 microglia cells retained
    v1 asked the subsample script for 200 cells, causing a 96.8% ATAC
    correlation filtering rate (too sparse).  This script writes all
    microglia cells; the subsample script is then called with
    --n-per-type 9999 to use all of them for Phase 3.

OUTPUTS (all under data/processed/seaad/):
  seaad_paired_rna_mg2k_hvg.h5ad    (n_mg cells × 2000 microglia HVGs, raw)
  seaad_paired_atac_mg10k_hvp.h5ad  (n_mg cells × 10000 microglia HVPs, raw)

After this script finishes, check the gene list in the log for AD-relevant
genes (SPI1, TREM2, TYROBP, CSF1R, MEF2C, STAT3, APOE, P2RY12) — then
proceed to subsample and Phase 3 regardless of which genes appear.

Run via:
    sbatch slurm/run_seaad_hvg_selection_microglia_v2.sh
"""

import gc
import logging
from pathlib import Path

import anndata as ad
import numpy as np

from screni.data.feature_selection import (
    filter_chr_peaks,
    select_variable_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
SEAAD_DIR    = Path("data/processed/seaad")
CELL_TYPE    = "Microglia-PVM"
N_HVGS       = 2_000     # FIX 2: was 500
N_HVPS       = 10_000

# Candidate obs column names for subclass — checked in order
SUBCLASS_CANDIDATES = ["Subclass", "subclass", "cell_type"]

# Candidate raw-count layer names in seaad_paired_rna.h5ad — checked in order
COUNT_LAYER_CANDIDATES = ["counts", "UMIs", "raw", "spliced"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sanitise_obs(obs):
    """Mirror _pull_obs from run_seaad_hvg_selection.py."""
    obs = obs.copy()
    if "cell_type" not in obs.columns:
        for c in SUBCLASS_CANDIDATES:
            if c in obs.columns:
                obs["cell_type"] = obs[c]
                break
    for c in obs.columns:
        if obs[c].dtype == "object":
            obs[c] = obs[c].astype(str)
    return obs


def _find_subclass_col(obs) -> str:
    for c in SUBCLASS_CANDIDATES:
        if c in obs.columns:
            return c
    raise KeyError(
        f"No subclass column found. Checked: {SUBCLASS_CANDIDATES}. "
        f"Available: {list(obs.columns)[:15]}"
    )


def _verify_integer(mat, label: str) -> bool:
    """Sample first 500 rows and check for integer values."""
    sample = mat[:min(500, mat.shape[0])]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    is_int = np.allclose(sample, np.round(sample))
    vmax   = float(sample.max())
    logger.info(f"    {label}: max={vmax:.2f}  integer={is_int}")
    return is_int and vmax > 10


# ─── RNA raw-count retrieval ──────────────────────────────────────────────────

def _load_rna_raw_counts(mg_mask: np.ndarray):
    """
    Returns (X_counts, obs, var) for microglia cells using the best available
    source of raw counts.

    Priority:
      1. seaad_paired_rna.h5ad — check for raw-count layers or integer .X
      2. seaad_paired_integrated.h5mu — confirmed raw, but 91 GB (needs 300G)
    """
    rna_path = SEAAD_DIR / "seaad_paired_rna.h5ad"

    # ── Attempt 1: seaad_paired_rna.h5ad ──────────────────────────────────────
    if rna_path.exists():
        logger.info(f"\nAttempting to read raw counts from {rna_path.name} ...")
        rna = ad.read_h5ad(rna_path, backed="r")
        logger.info(f"  shape: {rna.shape}")
        logger.info(f"  layers: {list(rna.layers.keys())}")
        sc = _find_subclass_col(rna.obs)
        # ensure mask matches this file's cell order
        mg_mask_here = (rna.obs[sc] == CELL_TYPE).values

        # Try each candidate layer
        for lname in COUNT_LAYER_CANDIDATES:
            if lname in rna.layers:
                logger.info(f"  Checking layers['{lname}'] ...")
                if _verify_integer(rna.layers[lname], f"layers['{lname}']"):
                    X = rna.layers[lname][mg_mask_here]
                    if hasattr(X, "toarray"):
                        X = X.toarray()
                    obs = _sanitise_obs(rna.obs[mg_mask_here])
                    var = rna.var.copy()
                    rna.file.close()
                    logger.info(f"  ✓ Using layers['{lname}'] as raw counts")
                    return X.astype(np.float32), obs, var

        # Try .X directly
        logger.info("  Checking .X ...")
        if _verify_integer(rna.X, ".X"):
            X = rna.X[mg_mask_here]
            if hasattr(X, "toarray"):
                X = X.toarray()
            obs = _sanitise_obs(rna.obs[mg_mask_here])
            var = rna.var.copy()
            rna.file.close()
            logger.info("  ✓ Using .X as raw counts")
            return X.astype(np.float32), obs, var

        rna.file.close()
        logger.warning(
            f"  ✗ {rna_path.name}: no raw counts found "
            f"(.X max ≤ 10 or non-integer). Falling back to h5mu."
        )
    else:
        logger.warning(f"  {rna_path.name} not found. Falling back to h5mu.")

    # ── Attempt 2: seaad_paired_integrated.h5mu ───────────────────────────────
    h5mu_path = SEAAD_DIR / "seaad_paired_integrated.h5mu"
    if not h5mu_path.exists():
        raise FileNotFoundError(
            f"Neither {rna_path.name} nor {h5mu_path.name} found. "
            f"Cannot proceed without raw RNA counts."
        )

    logger.info(f"\nLoading {h5mu_path.name} (91 GB — this takes 20-40 min) ...")
    import muon as mu
    mdata = mu.read(str(h5mu_path))
    mod   = mdata.mod["rna"]
    logger.info(f"  RNA mod shape: {mod.shape}")

    if "counts" not in mod.layers:
        raise KeyError(
            "seaad_paired_integrated.h5mu mod['rna'] has no layers['counts']. "
            "The h5mu may not have been produced by integrate_seaad_paired()."
        )

    sc      = _find_subclass_col(mod.obs)
    mg_mask = (mod.obs[sc] == CELL_TYPE).values
    X       = mod.layers["counts"][mg_mask]
    if hasattr(X, "toarray"):
        X = X.toarray()
    obs = _sanitise_obs(mod.obs[mg_mask])
    var = mod.var.copy()
    del mdata, mod
    gc.collect()
    logger.info(f"  ✓ Using h5mu layers['counts'] as raw counts")
    return X.astype(np.float32), obs, var


# ─── RNA HVG selection ────────────────────────────────────────────────────────

def select_microglia_rna():
    # ── 1. Get embedding from the existing HVG file (small, backed) ───────────
    hvg_ref_path = SEAAD_DIR / "seaad_paired_rna_hvg.h5ad"
    if not hvg_ref_path.exists():
        logger.warning(
            f"  {hvg_ref_path.name} not found — output will lack obsm['X_pca']. "
            f"KNN must be recomputed from scratch in the subsample step."
        )
        joint_emb    = None
        mg_obs_names = None
    else:
        logger.info(f"Reading embedding from {hvg_ref_path.name} (backed) ...")
        hvg_ref = ad.read_h5ad(hvg_ref_path, backed="r")
        sc      = _find_subclass_col(hvg_ref.obs)
        mg_mask_ref = (hvg_ref.obs[sc] == CELL_TYPE).values
        logger.info(f"  {mg_mask_ref.sum()} microglia cells in HVG reference")
        if "X_pca" in hvg_ref.obsm:
            joint_emb = np.asarray(hvg_ref.obsm["X_pca"])[mg_mask_ref].copy()
        else:
            logger.warning("  obsm['X_pca'] not found in HVG ref — KNN recomputed later")
            joint_emb = None
        mg_obs_names = hvg_ref.obs_names[mg_mask_ref].copy()
        hvg_ref.file.close()

    # ── 2. Load raw counts for microglia ──────────────────────────────────────
    sc_dummy_mask = None   # not used; _load_rna_raw_counts handles its own mask
    X_counts, obs, var = _load_rna_raw_counts(sc_dummy_mask)
    logger.info(f"\nMicroglia RNA: {X_counts.shape[0]} cells × {X_counts.shape[1]} genes")

    # ── 3. Align with embedding obs_names if available ────────────────────────
    if mg_obs_names is not None and not (obs.index == mg_obs_names).all():
        logger.warning(
            "  obs_names mismatch between raw RNA and HVG reference — "
            "embedding will NOT be attached. KNN recomputed in subsample step."
        )
        joint_emb = None

    # ── 4. Build AnnData ──────────────────────────────────────────────────────
    import anndata as ad_m
    rna_mg = ad_m.AnnData(
        X   = X_counts,
        obs = obs,
        var = var,
    )
    del X_counts
    gc.collect()

    if joint_emb is not None:
        rna_mg.obsm["X_pca"] = joint_emb

    # ── 5. FIX 2: Select 2000 microglia-specific HVGs ─────────────────────────
    logger.info(f"\nSelecting {N_HVGS} microglia-specific HVGs (Seurat v3 VST, raw counts) ...")
    rna_mg_hvg = select_variable_features(rna_mg, n_features=N_HVGS, span=0.3)
    del rna_mg
    gc.collect()

    logger.info(f"  Selected shape: {rna_mg_hvg.shape}")
    logger.info(f"\n  First 50 microglia HVGs (2000-gene selection, raw counts):")
    logger.info(f"  {list(rna_mg_hvg.var_names[:50])}")

    # ── 6. AD-relevant gene checklist ─────────────────────────────────────────
    logger.info("\n  === AD-relevant gene presence in 2000-HVG set ===")
    ad_genes = {
        # Master regulators / TFs
        "SPI1":   "Master myeloid TF (PU.1); constitutive in microglia",
        "STAT3":  "Reactive astrogliosis TF; GFAP promoter binding",
        "MEF2C":  "AD GWAS TF; regulates BDNF and synaptic genes",
        "RUNX1":  "Myeloid TF; chromatin remodelling in microglia",
        "NR1H3":  "LXRα; APOE/ABCA1 lipid regulatory axis",
        "IRF8":   "Interferon regulatory factor; microglia identity",
        # Microglia markers / signalling
        "TREM2":  "DAM receptor; signals to APOE program",
        "TYROBP": "TREM2 co-receptor (DAP12); SPI1 target",
        "CSF1R":  "Microglia survival receptor; SPI1 target",
        "P2RY12": "Homeostatic microglia marker (variable between states)",
        "CX3CR1": "Homeostatic microglia marker",
        "AIF1":   "IBA1; pan-microglia marker (SPI1 target)",
        # AD downstream targets
        "APOE":   "Lipid metabolism; TREM2-DAM upregulated",
        "GFAP":   "Reactive astrocyte marker; STAT3 target",
        "C1QA":   "Complement; microglia-expressed",
        "C1QB":   "Complement",
        "C1QC":   "Complement",
    }
    present  = [g for g in ad_genes if g in rna_mg_hvg.var_names]
    absent   = [g for g in ad_genes if g not in rna_mg_hvg.var_names]
    logger.info(f"  PRESENT ({len(present)}): {present}")
    logger.info(f"  ABSENT  ({len(absent)}):  {absent}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out = SEAAD_DIR / "seaad_paired_rna_mg2k_hvg.h5ad"
    rna_mg_hvg.write_h5ad(out)
    logger.info(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    del rna_mg_hvg


# ─── ATAC HVP selection ───────────────────────────────────────────────────────

def select_microglia_atac():
    atac_path = SEAAD_DIR / "seaad_paired_atac.h5ad"
    hvp_ref_path = SEAAD_DIR / "seaad_paired_atac_hvp.h5ad"

    logger.info(f"\n{'='*60}\n  ATAC HVP selection\n{'='*60}")

    # Get microglia mask from HVP reference (backed, small)
    if hvp_ref_path.exists():
        hvp_ref = ad.read_h5ad(hvp_ref_path, backed="r")
        sc      = _find_subclass_col(hvp_ref.obs)
        mg_mask = (hvp_ref.obs[sc] == CELL_TYPE).values
        logger.info(f"  {mg_mask.sum()} microglia cells in ATAC HVP reference")
        hvp_ref.file.close()
    else:
        # Fall back: find mask from RNA output we just wrote
        rna_out = SEAAD_DIR / "seaad_paired_rna_mg2k_hvg.h5ad"
        if rna_out.exists():
            rna_out_ref = ad.read_h5ad(rna_out, backed="r")
            mg_obs = set(rna_out_ref.obs_names)
            rna_out_ref.file.close()
            logger.info(f"  Using RNA output obs_names for ATAC alignment ({len(mg_obs)} cells)")
            mg_mask = None  # handled below
        else:
            raise FileNotFoundError(
                "Cannot determine microglia cell mask: neither "
                f"{hvp_ref_path.name} nor {rna_out.name} exists."
            )

    # Load full ATAC
    if not atac_path.exists():
        raise FileNotFoundError(f"Missing: {atac_path}")

    logger.info(f"Loading {atac_path.name} (15.7 GB — ~15 min) ...")
    atac_full = ad.read_h5ad(atac_path)
    logger.info(f"  shape: {atac_full.shape}")

    # Verify .X is raw counts (expected per integration history doc)
    if not _verify_integer(atac_full.X, "ATAC .X"):
        logger.warning(
            "  ATAC .X does not appear to be integer-valued Tn5 counts. "
            "Proceeding anyway — VST may produce suboptimal results."
        )

    # Apply mask
    if mg_mask is not None:
        sc = _find_subclass_col(atac_full.obs)
        mg_mask_atac = (atac_full.obs[sc] == CELL_TYPE).values
    else:
        mg_mask_atac = atac_full.obs_names.isin(mg_obs)
    logger.info(f"  Microglia ATAC cells: {mg_mask_atac.sum()}")

    X_atac = atac_full.X[mg_mask_atac]
    if hasattr(X_atac, "toarray"):
        X_atac = X_atac.toarray()
    obs_atac = _sanitise_obs(atac_full.obs[mg_mask_atac])
    var_atac = atac_full.var.copy()
    del atac_full
    gc.collect()

    import anndata as ad_m
    atac_mg = ad_m.AnnData(
        X   = X_atac.astype(np.float32),
        obs = obs_atac,
        var = var_atac,
    )
    del X_atac
    gc.collect()

    # Chr filter then HVP selection
    atac_mg = filter_chr_peaks(atac_mg)
    logger.info(f"\nSelecting {N_HVPS} microglia-specific HVPs (Seurat v3 VST, span=0.5) ...")
    atac_mg_hvp = select_variable_features(atac_mg, n_features=N_HVPS, span=0.5)
    del atac_mg
    gc.collect()
    logger.info(f"  Selected shape: {atac_mg_hvp.shape}")

    out = SEAAD_DIR / "seaad_paired_atac_mg10k_hvp.h5ad"
    atac_mg_hvp.write_h5ad(out)
    logger.info(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    del atac_mg_hvp


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info(
        f"=== Microglia HVG/HVP selection v2 ===\n"
        f"  FIXES: raw counts, {N_HVGS} HVGs, all {CELL_TYPE} cells\n"
    )
    select_microglia_rna()
    gc.collect()
    select_microglia_atac()

    logger.info(
        "\n=== Done ===\n"
        "Next steps:\n"
        "  1. Check the AD-relevant gene presence list above.\n"
        f"  2. Run subsample (uses ALL {CELL_TYPE} cells):\n"
        "       python scripts/subsample_seaad_paired.py \\\n"
        "           --seed 88 \\\n"
        "           --n-per-type 9999 \\\n"
        "           --cell-types Microglia-PVM \\\n"
        "           --rna  data/processed/seaad/seaad_paired_rna_mg2k_hvg.h5ad \\\n"
        "           --atac data/processed/seaad/seaad_paired_atac_mg10k_hvp.h5ad\n"
        "  3. Run Phase 3 (picks up sub88 via glob):\n"
        "       sbatch slurm/run_gene_peak.sh\n"
        "  4. Validate:\n"
        "       python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub88\n"
    )


if __name__ == "__main__":
    main()
