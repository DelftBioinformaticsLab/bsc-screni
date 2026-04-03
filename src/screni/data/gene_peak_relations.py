"""Phase 3: Gene-peak-TF relationship inference.

Establishes regulatory triplets (TF → peak → target gene) that constrain
the random forest model in wScReNI.

Matches the original R functions:
    - ``peak_gene_overlap_GR1()``: genomic proximity (250kb window)
    - ``gene_peak_corr1()``: Spearman correlation filtering
    - ``Get_motif_peak_pair_df0()``: TF motif matching in peaks
    - ``peak_gene_TF_match()``: triplet assembly
    - ``peakMat()`` / ``peak_gene_TF_labs()``: RF input preparation

Key parameters from the paper:
    - TSS window: 250kb upstream + 250kb downstream
    - Correlation threshold: |Spearman r| > 0.1
    - Motif p-value cutoff: 5^(-4) = 0.0016
    - Gaussian noise for peak matrix: N(0, 10^-5)
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from screni.data.utils import parse_peak_name, peaks_to_dataframe

logger = logging.getLogger(__name__)

# Default parameters matching the paper
DEFAULT_UPSTREAM_BP = 250_000
DEFAULT_DOWNSTREAM_BP = 250_000
DEFAULT_CORR_THRESHOLD = 0.1
DEFAULT_MOTIF_PVALUE = 5**-4  # 0.0016
DEFAULT_NOISE_SD = 1e-5


# =========================================================================
#  3a. Peak-gene overlap (genomic proximity)
# =========================================================================


def load_gene_annotations(gtf_path: Path | str) -> pd.DataFrame:
    """Load gene annotations from an Ensembl GTF file.

    Extracts gene-level records with TSS coordinates.

    Returns
    -------
    DataFrame with columns: Chromosome, Start, End, Strand, gene_name, gene_id.
    """
    import gzip

    gtf_path = Path(gtf_path)
    logger.info(f"Loading gene annotations from {gtf_path.name}...")

    records = []
    opener = gzip.open if gtf_path.suffix == ".gz" else open

    with opener(str(gtf_path), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue

            chrom = parts[0]
            # Ensembl GTFs may not have 'chr' prefix
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"

            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]

            # Parse attributes
            attrs = parts[8]
            gene_name = _parse_gtf_attr(attrs, "gene_name")
            gene_id = _parse_gtf_attr(attrs, "gene_id")

            if gene_name:
                records.append({
                    "Chromosome": chrom,
                    "Start": start,
                    "End": end,
                    "Strand": strand,
                    "gene_name": gene_name,
                    "gene_id": gene_id or "",
                })

    df = pd.DataFrame(records)
    logger.info(f"  Loaded {len(df)} gene annotations")
    return df


def _parse_gtf_attr(attr_string: str, key: str) -> str | None:
    """Extract a value from a GTF attribute string."""
    for attr in attr_string.split(";"):
        attr = attr.strip()
        if attr.startswith(f'{key} "'):
            return attr.split('"')[1]
    return None


def find_peak_gene_overlaps(
    gene_annotations: pd.DataFrame,
    peak_names: list[str] | np.ndarray,
    hvg_names: list[str] | np.ndarray,
    upstream_bp: int = DEFAULT_UPSTREAM_BP,
    downstream_bp: int = DEFAULT_DOWNSTREAM_BP,
) -> pd.DataFrame:
    """Find peaks within a window around each gene's TSS.

    Matches ``peak_gene_overlap_GR1()`` from the original R code.

    Parameters
    ----------
    gene_annotations
        Gene annotations with Chromosome, Start, End, Strand, gene_name.
    peak_names
        ATAC peak names (format: chrX:start-end).
    hvg_names
        Highly variable gene names to restrict to.
    upstream_bp
        Window upstream of TSS (default: 250,000).
    downstream_bp
        Window downstream of TSS (default: 250,000).

    Returns
    -------
    DataFrame with columns: gene, peak (pairs in genomic proximity).
    """
    logger.info(
        f"Finding peak-gene overlaps (window: {upstream_bp // 1000}kb up, "
        f"{downstream_bp // 1000}kb down)..."
    )

    # Filter gene annotations to HVGs
    gene_ann = gene_annotations[
        gene_annotations["gene_name"].isin(hvg_names)
    ].copy().reset_index(drop=True)
    logger.info(f"  Genes with annotations: {len(gene_ann)} / {len(hvg_names)}")

    # Compute TSS and extend
    tss = gene_ann["Start"].copy()
    minus_mask = gene_ann["Strand"] == "-"
    tss[minus_mask] = gene_ann.loc[minus_mask, "End"]

    gene_ann = gene_ann.assign(
        tss_start=np.maximum(0, tss - upstream_bp).astype(int),
        tss_end=(tss + downstream_bp).astype(int),
    )

    # Parse peak coordinates
    peak_df = peaks_to_dataframe(peak_names)
    logger.info(f"  Peaks parsed: {len(peak_df)} / {len(peak_names)}")

    if len(peak_df) == 0:
        logger.warning("  No peaks could be parsed!")
        return pd.DataFrame(columns=["gene", "peak"])

    # Find overlaps: for each chromosome, check which peaks fall within
    # each gene's extended TSS window. This is a simple interval overlap.
    results = []
    for chrom in gene_ann["Chromosome"].unique():
        genes_chr = gene_ann[gene_ann["Chromosome"] == chrom]
        peaks_chr = peak_df[peak_df["Chromosome"] == chrom]

        if len(peaks_chr) == 0:
            continue

        # For each gene on this chromosome, find overlapping peaks
        peak_starts = peaks_chr["Start"].values
        peak_ends = peaks_chr["End"].values
        peak_names_chr = peaks_chr["Name"].values

        for _, gene_row in genes_chr.iterrows():
            g_start = gene_row["tss_start"]
            g_end = gene_row["tss_end"]
            # Overlap condition: peak_start < gene_end AND peak_end > gene_start
            overlap_mask = (peak_starts < g_end) & (peak_ends > g_start)
            for pname in peak_names_chr[overlap_mask]:
                results.append({"gene": gene_row["gene_name"], "peak": pname})

    result = pd.DataFrame(results).drop_duplicates()

    if len(result) == 0:
        logger.warning("  No overlaps found! Check chromosome naming consistency.")
        return result

    # Stats
    peaks_per_gene = result.groupby("gene").size()
    logger.info(
        f"  Found {len(result)} gene-peak proximity pairs "
        f"({result['gene'].nunique()} genes, {result['peak'].nunique()} peaks)\n"
        f"  Peaks per gene: median={peaks_per_gene.median():.0f}, "
        f"mean={peaks_per_gene.mean():.0f}, "
        f"range=[{peaks_per_gene.min()}, {peaks_per_gene.max()}]"
    )

    return result


# =========================================================================
#  3b. Gene-peak correlation
# =========================================================================


def filter_by_correlation(
    overlap_pairs: pd.DataFrame,
    rna_adata: ad.AnnData,
    atac_adata: ad.AnnData,
    threshold: float = DEFAULT_CORR_THRESHOLD,
    min_nonzero_cells: int = 5,
) -> pd.DataFrame:
    """Filter gene-peak pairs by Spearman correlation.

    Matches ``gene_peak_corr1()`` from the original R code.

    For each (gene, peak) pair, computes Spearman correlation between
    gene expression and peak accessibility across cells where both are
    non-zero. Keeps pairs with |correlation| > threshold.

    Parameters
    ----------
    overlap_pairs
        DataFrame with columns: gene, peak.
    rna_adata
        RNA AnnData (matched cells, same order as atac_adata).
    atac_adata
        ATAC AnnData (matched cells).
    threshold
        Minimum absolute Spearman correlation (default: 0.1).
    min_nonzero_cells
        Minimum number of cells with both gene and peak non-zero
        to compute correlation.

    Returns
    -------
    DataFrame with columns: gene, peak, spearman_r.
    """
    logger.info(
        f"Filtering {len(overlap_pairs)} gene-peak pairs by correlation "
        f"(threshold: |r| > {threshold})..."
    )

    # Build lookup indices
    gene_to_idx = {g: i for i, g in enumerate(rna_adata.var_names)}
    peak_to_idx = {p: i for i, p in enumerate(atac_adata.var_names)}

    results = []
    n_skipped_missing = 0
    n_skipped_sparse = 0

    for _, row in overlap_pairs.iterrows():
        gene, peak = row["gene"], row["peak"]

        if gene not in gene_to_idx or peak not in peak_to_idx:
            n_skipped_missing += 1
            continue

        gi = gene_to_idx[gene]
        pi = peak_to_idx[peak]

        # Get expression/accessibility vectors
        if sp.issparse(rna_adata.X):
            gene_expr = np.array(rna_adata.X[:, gi].todense()).flatten()
        else:
            gene_expr = rna_adata.X[:, gi].flatten()

        if sp.issparse(atac_adata.X):
            peak_acc = np.array(atac_adata.X[:, pi].todense()).flatten()
        else:
            peak_acc = atac_adata.X[:, pi].flatten()

        # Use only cells where both are non-zero
        both_nonzero = (gene_expr > 0) & (peak_acc > 0)
        n_valid = both_nonzero.sum()

        if n_valid < min_nonzero_cells:
            n_skipped_sparse += 1
            continue

        corr, _ = spearmanr(gene_expr[both_nonzero], peak_acc[both_nonzero])

        if np.isnan(corr):
            continue

        if abs(corr) > threshold:
            results.append({"gene": gene, "peak": peak, "spearman_r": corr})

    result_df = pd.DataFrame(results)
    logger.info(
        f"  Kept {len(result_df)} correlated pairs "
        f"(skipped {n_skipped_missing} missing, {n_skipped_sparse} too sparse)\n"
        f"  Filtering rate: {1 - len(result_df) / max(1, len(overlap_pairs)):.1%} removed"
    )

    if len(result_df) > 0:
        logger.info(
            f"  Correlation stats: "
            f"mean |r| = {result_df['spearman_r'].abs().mean():.3f}, "
            f"positive: {(result_df['spearman_r'] > 0).sum()}, "
            f"negative: {(result_df['spearman_r'] < 0).sum()}"
        )

    return result_df


# =========================================================================
#  3c. Motif matching
# =========================================================================


def load_jaspar_motifs(
    jaspar_dir: Path | str,
) -> list[tuple[str, str, list[list[float]]]]:
    """Load JASPAR PFM motifs from a directory of .pfm or .jaspar files.

    Can also load a single concatenated JASPAR-format file.

    Returns list of (motif_id, tf_name, pfm_matrix) tuples.
    Each pfm_matrix is a list of 4 lists (A, C, G, T counts per position).
    """
    jaspar_dir = Path(jaspar_dir)
    motifs = []

    if jaspar_dir.is_file():
        files = [jaspar_dir]
    else:
        files = sorted(jaspar_dir.glob("*.jaspar")) + sorted(jaspar_dir.glob("*.pfm"))

    for fpath in files:
        with open(fpath) as f:
            current_id = None
            current_name = None
            current_matrix: list[list[float]] = []

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous motif
                    if current_id and len(current_matrix) == 4:
                        motifs.append((current_id, current_name or current_id, current_matrix))
                    # Parse header: >MA0001.1 ARNT
                    parts = line[1:].split()
                    current_id = parts[0] if parts else "unknown"
                    current_name = parts[1] if len(parts) > 1 else current_id
                    current_matrix = []
                elif line and not line.startswith("#"):
                    # Parse row of counts: A [ 1 2 3 4 ]
                    # Remove letter prefix and brackets
                    cleaned = line.replace("[", "").replace("]", "")
                    # Remove single-letter prefix if present (e.g., "A  1 2 3")
                    tokens = cleaned.split()
                    if tokens and tokens[0].isalpha() and len(tokens[0]) == 1:
                        tokens = tokens[1:]
                    try:
                        row = [float(x) for x in tokens]
                        if row:
                            current_matrix.append(row)
                    except ValueError:
                        continue

            # Don't forget last motif
            if current_id and len(current_matrix) == 4:
                motifs.append((current_id, current_name or current_id, current_matrix))

    logger.info(f"  Loaded {len(motifs)} JASPAR motifs from {jaspar_dir}")
    return motifs


def _pfm_to_pwm(pfm: list[list[float]], pseudocount: float = 0.01) -> list[list[float]]:
    """Convert a position frequency matrix to a position weight matrix (log-odds).

    Parameters
    ----------
    pfm
        4 x N matrix of counts (A, C, G, T).
    pseudocount
        Added to each count to avoid log(0).

    Returns
    -------
    4 x N log-odds PWM.
    """
    n_pos = len(pfm[0])
    pwm = []
    for base_idx in range(4):
        row = []
        for pos in range(n_pos):
            total = sum(pfm[b][pos] + pseudocount for b in range(4))
            freq = (pfm[base_idx][pos] + pseudocount) / total
            row.append(np.log2(freq / 0.25))
        pwm.append(row)
    return pwm


def _scan_sequence_with_pwm(
    seq: str,
    pwm: np.ndarray,
    threshold: float,
) -> list[tuple[int, float]]:
    """Scan a DNA sequence with a PWM and return positions above threshold.

    Pure numpy implementation - no C extensions needed.

    Parameters
    ----------
    seq
        DNA sequence (uppercase, ACGT only).
    pwm
        4 x W numpy array (rows = A, C, G, T; columns = positions).
    threshold
        Minimum score to report a hit.

    Returns
    -------
    List of (position, score) tuples.
    """
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    w = pwm.shape[1]

    if len(seq) < w:
        return []

    # Encode sequence as integer array
    encoded = np.array([base_to_idx.get(b, -1) for b in seq], dtype=np.int8)

    hits = []
    for i in range(len(seq) - w + 1):
        window = encoded[i : i + w]
        if np.any(window < 0):  # Skip windows with N's
            continue
        score = sum(pwm[window[j], j] for j in range(w))
        if score >= threshold:
            hits.append((i, score))

    return hits


def _estimate_pwm_threshold(pwm: np.ndarray, pvalue: float) -> float:
    """Estimate a PWM score threshold for a given p-value.

    Uses the max-column approximation: for each position, takes the
    distribution of scores and estimates the threshold from the total
    score distribution assuming independence between positions.

    For a more precise threshold, we'd need the full score distribution
    (dynamic programming), but this approximation is sufficient and matches
    the spirit of motifmatchr's approach.
    """
    w = pwm.shape[1]
    # Maximum possible score
    max_score = np.sum(np.max(pwm, axis=0))
    # Minimum possible score
    min_score = np.sum(np.min(pwm, axis=0))
    # Mean score under uniform background
    mean_score = np.sum(np.mean(pwm, axis=0))
    # Approximate SD under uniform background
    var_per_pos = np.var(pwm, axis=0)
    sd_score = np.sqrt(np.sum(var_per_pos))

    if sd_score == 0:
        return max_score

    # Use normal approximation: threshold = mean + z * sd
    from scipy.stats import norm
    z = norm.ppf(1 - pvalue)
    threshold = mean_score + z * sd_score

    # Clamp to valid range
    return float(np.clip(threshold, min_score, max_score))


def match_motifs_to_peaks(
    peak_names: list[str] | np.ndarray,
    genome_fasta: Path | str,
    jaspar_motifs_path: Path | str,
    pvalue_cutoff: float = DEFAULT_MOTIF_PVALUE,
) -> pd.DataFrame:
    """Scan ATAC peaks for TF binding motifs.

    Matches ``Get_motif_peak_pair_df0()`` from the original R code.
    Uses a pure-numpy PWM scanner (no C extensions required).

    Parameters
    ----------
    peak_names
        ATAC peak names (format: chrX:start-end).
    genome_fasta
        Path to genome FASTA file (mm10 or hg38).
    jaspar_motifs_path
        Path to JASPAR motif file(s) (.jaspar or directory of .pfm files).
    pvalue_cutoff
        P-value threshold for motif matching (default: 5^-4 = 0.0016).

    Returns
    -------
    DataFrame with columns: TF, peak, motif_id, score.
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError(
            "pyfaidx is required for genome sequence extraction. "
            "Install with: pip install pyfaidx"
        )

    genome_fasta = Path(genome_fasta)
    logger.info(
        f"Scanning {len(peak_names)} peaks for motifs "
        f"(p-value cutoff: {pvalue_cutoff})..."
    )

    # Load genome
    genome = Fasta(str(genome_fasta))

    # Load JASPAR motifs and convert to PWMs
    jaspar_motifs = load_jaspar_motifs(jaspar_motifs_path)
    if not jaspar_motifs:
        raise ValueError(f"No motifs loaded from {jaspar_motifs_path}")

    # Build PWM arrays and thresholds
    motif_data = []  # (motif_id, tf_name, pwm_fwd, pwm_rc, threshold)
    for motif_id, tf_name, pfm in jaspar_motifs:
        pwm = np.array(_pfm_to_pwm(pfm))  # shape: (4, W)
        threshold = _estimate_pwm_threshold(pwm, pvalue_cutoff)
        # Reverse complement: reverse columns, swap A<->T and C<->G (rows 0<->3, 1<->2)
        pwm_rc = pwm[[3, 2, 1, 0], ::-1]
        motif_data.append((motif_id, tf_name, pwm, pwm_rc, threshold))

    logger.info(f"  Prepared {len(motif_data)} motifs with score thresholds")

    # Scan each peak sequence
    results = []
    n_scanned = 0

    for peak in peak_names:
        try:
            chrom, start, end = parse_peak_name(peak)
        except ValueError:
            continue

        if chrom not in genome:
            continue

        chrom_len = len(genome[chrom])
        start = max(0, start)
        end = min(end, chrom_len)
        seq = str(genome[chrom][start:end]).upper()

        if len(seq) == 0:
            continue

        for motif_id, tf_name, pwm_fwd, pwm_rc, threshold in motif_data:
            # Scan forward strand
            hits_fwd = _scan_sequence_with_pwm(seq, pwm_fwd, threshold)
            # Scan reverse complement
            hits_rc = _scan_sequence_with_pwm(seq, pwm_rc, threshold)

            best_score = -np.inf
            if hits_fwd:
                best_score = max(best_score, max(s for _, s in hits_fwd))
            if hits_rc:
                best_score = max(best_score, max(s for _, s in hits_rc))

            if best_score > -np.inf:
                results.append({
                    "TF": tf_name,
                    "peak": peak,
                    "motif_id": motif_id,
                    "score": best_score,
                })

        n_scanned += 1
        if n_scanned % 1000 == 0:
            logger.info(f"  Scanned {n_scanned} / {len(peak_names)} peaks...")

    logger.info(f"  Scanned {n_scanned} peaks total")

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.drop_duplicates(subset=["TF", "peak"], keep="first")

    logger.info(
        f"  Found {len(result_df)} TF-peak matches "
        f"({result_df['TF'].nunique() if len(result_df) > 0 else 0} unique TFs, "
        f"{result_df['peak'].nunique() if len(result_df) > 0 else 0} unique peaks)"
    )

    if len(result_df) > 0:
        matches_per_peak = result_df.groupby("peak").size()
        logger.info(
            f"  Motifs per peak: median={matches_per_peak.median():.0f}, "
            f"mean={matches_per_peak.mean():.0f}"
        )

    return result_df


# =========================================================================
#  3d. TF-peak-gene triplet assembly
# =========================================================================


def assemble_triplets(
    correlated_pairs: pd.DataFrame,
    motif_matches: pd.DataFrame,
    hvg_names: list[str] | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine correlated gene-peak pairs with motif matches into triplets.

    Matches ``peak_gene_TF_match()`` from the original R code.

    Parameters
    ----------
    correlated_pairs
        DataFrame with columns: gene, peak, spearman_r.
    motif_matches
        DataFrame with columns: TF, peak, motif_id.
    hvg_names
        List of all HVG names (to determine which TFs are in our gene set).

    Returns
    -------
    Tuple of:
        - triplets: DataFrame with columns: TF, peak, target_gene, spearman_r
        - gene_labels: DataFrame with columns: gene, type (TF/target),
          associated_peaks (comma-separated), associated_TFs (comma-separated)
    """
    logger.info("Assembling TF-peak-gene triplets...")

    hvg_set = set(hvg_names)

    # Inner join: correlate gene-peak pairs with motif matches
    # Only keep triplets where the TF is also in our HVG set
    motif_in_hvg = motif_matches[motif_matches["TF"].isin(hvg_set)].copy()
    logger.info(
        f"  TFs in HVG set: {motif_in_hvg['TF'].nunique()} / "
        f"{motif_matches['TF'].nunique()} total TFs"
    )

    triplets = correlated_pairs.merge(
        motif_in_hvg[["TF", "peak", "motif_id"]],
        on="peak",
        how="inner",
    )
    triplets = triplets.rename(columns={"gene": "target_gene"})

    # Remove self-regulation (TF == target)
    triplets = triplets[triplets["TF"] != triplets["target_gene"]].copy()

    logger.info(
        f"  Assembled {len(triplets)} triplets "
        f"({triplets['TF'].nunique()} TFs → {triplets['target_gene'].nunique()} targets "
        f"via {triplets['peak'].nunique()} peaks)"
    )

    # Build gene labels
    tf_genes = set(triplets["TF"].unique())
    target_genes = set(hvg_names) - tf_genes

    # For each gene, collect associated peaks and TFs
    gene_peak_map = (
        correlated_pairs.groupby("gene")["peak"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .to_dict()
    )

    gene_tf_map = (
        triplets.groupby("target_gene")["TF"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .to_dict()
    )

    gene_labels = []
    for gene in hvg_names:
        gene_labels.append({
            "gene": gene,
            "type": "TF" if gene in tf_genes else "target",
            "associated_peaks": gene_peak_map.get(gene, ""),
            "associated_TFs": gene_tf_map.get(gene, ""),
        })

    gene_labels_df = pd.DataFrame(gene_labels)

    n_tf = (gene_labels_df["type"] == "TF").sum()
    n_target = (gene_labels_df["type"] == "target").sum()
    logger.info(f"  Gene labels: {n_tf} TFs ({n_tf/len(gene_labels_df):.1%}), {n_target} targets")

    return triplets, gene_labels_df


# =========================================================================
#  3e. Prepare matrices for RF input
# =========================================================================


def prepare_rf_inputs(
    atac_adata: ad.AnnData,
    correlated_pairs: pd.DataFrame,
    gene_labels: pd.DataFrame,
    noise_sd: float = DEFAULT_NOISE_SD,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Prepare the peak overlap matrix and gene labels for RF input.

    Matches ``peakMat()`` and ``peak_gene_TF_labs()`` from the original R code.

    For each target gene, extracts the accessibility values of its associated
    peaks and adds Gaussian noise to break ties.

    Parameters
    ----------
    atac_adata
        ATAC AnnData with subsampled cells and selected peaks.
    correlated_pairs
        DataFrame with columns: gene, peak, spearman_r.
    gene_labels
        DataFrame with columns: gene, type, associated_peaks.
    noise_sd
        Standard deviation of Gaussian noise (default: 10^-5).
    seed
        Random seed for noise generation.

    Returns
    -------
    Tuple of:
        - peak_matrix: ndarray of shape (n_cells, n_unique_peaks) with noise
        - peak_info: DataFrame mapping peak columns to genes
    """
    logger.info("Preparing RF input matrices...")

    rng = np.random.RandomState(seed)

    # Get all unique peaks from correlated pairs
    unique_peaks = sorted(correlated_pairs["peak"].unique())
    peak_to_idx = {p: i for i, p in enumerate(unique_peaks)}
    atac_peak_to_idx = {p: i for i, p in enumerate(atac_adata.var_names)}

    # Extract peak accessibility matrix
    valid_peaks = [p for p in unique_peaks if p in atac_peak_to_idx]
    valid_indices = [atac_peak_to_idx[p] for p in valid_peaks]

    if sp.issparse(atac_adata.X):
        peak_matrix = atac_adata.X[:, valid_indices].toarray().astype(np.float64)
    else:
        peak_matrix = atac_adata.X[:, valid_indices].astype(np.float64)

    # Add Gaussian noise
    noise = rng.normal(0, noise_sd, peak_matrix.shape)
    peak_matrix += noise

    logger.info(
        f"  Peak overlap matrix: {peak_matrix.shape} "
        f"(noise SD: {noise_sd})"
    )

    # Build peak info mapping
    peak_gene_groups = correlated_pairs.groupby("peak")["gene"].apply(list).to_dict()
    peak_info = pd.DataFrame({
        "peak": valid_peaks,
        "associated_genes": [
            ",".join(peak_gene_groups.get(p, []))
            for p in valid_peaks
        ],
    })

    logger.info(f"  Peak info: {len(peak_info)} peaks mapped to genes")

    return peak_matrix, peak_info


# =========================================================================
#  Convenience: run full Phase 3
# =========================================================================


def run_phase3(
    rna_adata: ad.AnnData,
    atac_adata: ad.AnnData,
    gene_annotations: pd.DataFrame,
    genome_fasta: Path | str | None = None,
    jaspar_motifs_path: Path | str | None = None,
    upstream_bp: int = DEFAULT_UPSTREAM_BP,
    downstream_bp: int = DEFAULT_DOWNSTREAM_BP,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    motif_pvalue: float = DEFAULT_MOTIF_PVALUE,
    output_dir: Path | str | None = None,
    prefix: str = "retinal",
) -> dict:
    """Run the full Phase 3 pipeline.

    Parameters
    ----------
    rna_adata
        Subsampled RNA AnnData (400 cells x 500 HVGs, raw counts).
    atac_adata
        Subsampled ATAC AnnData (400 cells x 10,000 peaks, raw counts).
    gene_annotations
        Gene body coordinates from GTF.
    genome_fasta
        Path to genome FASTA (required for motif matching, optional otherwise).
    jaspar_motifs_path
        Path to JASPAR motif file(s) (required for motif matching).
    upstream_bp, downstream_bp
        TSS window for peak-gene overlap.
    corr_threshold
        Minimum |Spearman r| for gene-peak pairs.
    motif_pvalue
        P-value cutoff for motif matching.
    output_dir
        If given, save intermediate results here.
    prefix
        Filename prefix for outputs (e.g., 'retinal' or 'pbmc').

    Returns
    -------
    Dict with keys: overlap_pairs, correlated_pairs, motif_matches,
    triplets, gene_labels, peak_matrix, peak_info.
    """
    logger.info(f"=== Phase 3: Gene-Peak-TF Relationships ({prefix}) ===")

    hvg_names = rna_adata.var_names.tolist()
    peak_names = atac_adata.var_names.tolist()

    # 3a: Peak-gene overlap
    overlap_pairs = find_peak_gene_overlaps(
        gene_annotations,
        peak_names,
        hvg_names,
        upstream_bp=upstream_bp,
        downstream_bp=downstream_bp,
    )

    # 3b: Correlation filtering
    correlated_pairs = filter_by_correlation(
        overlap_pairs,
        rna_adata,
        atac_adata,
        threshold=corr_threshold,
    )

    # 3c: Motif matching (requires genome FASTA + JASPAR motifs)
    if genome_fasta is not None and jaspar_motifs_path is not None:
        motif_matches = match_motifs_to_peaks(
            peak_names,
            genome_fasta,
            jaspar_motifs_path,
            pvalue_cutoff=motif_pvalue,
        )
    else:
        logger.warning(
            "  Skipping motif matching (genome FASTA or JASPAR motifs not provided). "
            "Triplet assembly will be incomplete."
        )
        motif_matches = pd.DataFrame(columns=["TF", "peak", "motif_id", "score"])

    # 3d: Triplet assembly
    triplets, gene_labels = assemble_triplets(
        correlated_pairs,
        motif_matches,
        hvg_names,
    )

    # 3e: Prepare RF inputs
    peak_matrix, peak_info = prepare_rf_inputs(
        atac_adata,
        correlated_pairs,
        gene_labels,
    )

    results = {
        "overlap_pairs": overlap_pairs,
        "correlated_pairs": correlated_pairs,
        "motif_matches": motif_matches,
        "triplets": triplets,
        "gene_labels": gene_labels,
        "peak_matrix": peak_matrix,
        "peak_info": peak_info,
    }

    # Save if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        correlated_pairs.to_csv(
            output_dir / f"{prefix}_peak_gene_pairs.csv", index=False
        )
        motif_matches.to_csv(
            output_dir / f"{prefix}_motif_peak_pairs.csv", index=False
        )
        triplets.to_csv(output_dir / f"{prefix}_triplets.csv", index=False)
        gene_labels.to_csv(output_dir / f"{prefix}_gene_labels.csv", index=False)
        np.savez_compressed(
            output_dir / f"{prefix}_peak_overlap_matrix.npz",
            peak_matrix=peak_matrix,
        )
        peak_info.to_csv(output_dir / f"{prefix}_peak_info.csv", index=False)

        logger.info(f"  Saved Phase 3 outputs to {output_dir}")

    return results
