"""Phase 3: Gene-peak-TF relationship inference.

Establishes regulatory triplets (TF -> peak -> target gene) that constrain
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

from screni.data.utils import load_gene_annotations, parse_peak_name, peaks_to_dataframe

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

    # Compute TSS and extend window.
    # For plus-strand genes: TSS = Start, upstream is lower coords.
    # For minus-strand genes: TSS = End, upstream is higher coords.
    # R code: start = ifelse(strand=='-', end-downstream, start-upstream)
    #          end   = ifelse(strand=='-', end+upstream,  start+downstream)
    plus_mask = gene_ann["Strand"] != "-"
    minus_mask = ~plus_mask

    tss_start = pd.Series(0, index=gene_ann.index, dtype=int)
    tss_end = pd.Series(0, index=gene_ann.index, dtype=int)

    tss_start[plus_mask] = gene_ann.loc[plus_mask, "Start"] - upstream_bp
    tss_end[plus_mask] = gene_ann.loc[plus_mask, "Start"] + downstream_bp

    tss_start[minus_mask] = gene_ann.loc[minus_mask, "End"] - downstream_bp
    tss_end[minus_mask] = gene_ann.loc[minus_mask, "End"] + upstream_bp

    gene_ann = gene_ann.assign(
        tss_start=np.maximum(0, tss_start).astype(int),
        tss_end=tss_end.astype(int),
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
    min_nonzero_cells: int = 2,
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

    # Detect binary ATAC data.  When peaks are strictly 0/1, the
    # "both-nonzero" filter makes the peak column constant (all 1s),
    # so Spearman is undefined.  Use all cells instead — zeros in
    # binary accessibility are informative, not technical dropout.
    if sp.issparse(atac_adata.X):
        atac_vals = atac_adata.X.data
    else:
        atac_vals = atac_adata.X[atac_adata.X > 0]
    atac_is_binary = np.all(atac_vals == 1.0)
    if atac_is_binary:
        logger.info("  ATAC data is binary -- using all cells for correlation")

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

        if atac_is_binary:
            # Binary ATAC: correlate across all cells (point-biserial)
            if peak_acc.sum() < min_nonzero_cells or gene_expr.sum() == 0:
                n_skipped_sparse += 1
                continue
            corr, _ = spearmanr(gene_expr, peak_acc)
        else:
            # Count ATAC: use only cells where both are non-zero
            # (matches R's gene_peak_corr1 which avoids dropout zeros)
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

    if results:
        result_df = pd.DataFrame(results)
    else:
        result_df = pd.DataFrame(columns=["gene", "peak", "spearman_r"])
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


def load_transfac_motifs(
    rds_path: Path | str,
    motif_db_path: Path | str,
    gene_name_type: str = "symbol",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load TRANSFAC PWMs and motif-to-TF mapping from the paper's files.

    Parameters
    ----------
    rds_path
        Path to ``all_motif_pwm.rds`` (R PWMatrixList).
    motif_db_path
        Path to ``Tranfac201803_*_MotifTFsFinal.txt`` (TSV mapping).
    gene_name_type
        Which column to use as TF name: 'symbol' uses the TFs column,
        'id' uses EnsemblID.

    Returns
    -------
    Tuple of:
        - pwm_dict: {accession: 4xN numpy array} (rows = A, C, G, T)
        - motif_db: DataFrame with columns Accession, ID, Name, TFs, EnsemblID
    """
    import warnings

    # Load motif database.  Two formats:
    #   Mouse: plain TSV (tab-separated, no quotes)
    #   Human: R-style write.table output (space-separated, quoted, row numbers)
    motif_db_path = Path(motif_db_path)
    with open(motif_db_path) as f:
        first_line = f.readline()

    if "\t" in first_line:
        # Tab-separated (mouse)
        motif_db = pd.read_csv(motif_db_path, sep="\t")
    else:
        # R-style write.table: "rownum" "Accession" "ID" "Name" "TFs" "EnsemblID"
        # Split on '" "' to handle semicolons in TF fields.
        rows = []
        with open(motif_db_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = [p.strip().strip('"') for p in line.strip().split('" "')]
                # Drop leading row-number (first part is always a digit)
                if parts and parts[0].isdigit():
                    parts = parts[1:]
                if len(parts) >= 5:
                    rows.append(parts[:5])
        motif_db = pd.DataFrame(rows, columns=["Accession", "ID", "Name", "TFs", "EnsemblID"])

    logger.info(f"  Loaded {len(motif_db)} motif-TF mappings from {motif_db_path.name}")

    # Load PWMs from RDS
    rds_path = Path(rds_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import rdata
        parsed = rdata.read_rds(str(rds_path))

    pwm_dict = {}
    for accession, pwm_obj in parsed.listData.items():
        # profileMatrix is an xarray DataArray with coords dim_0 = [A, C, G, T]
        mat = pwm_obj.profileMatrix.values  # shape (4, N)
        pwm_dict[str(accession)] = mat.astype(np.float64)

    logger.info(f"  Loaded {len(pwm_dict)} PWMs from {rds_path.name}")

    return pwm_dict, motif_db


def _select_motifs(
    motif_db: pd.DataFrame,
    hvg_names: list[str] | set[str],
    gene_name_type: str = "symbol",
) -> pd.DataFrame:
    """Pre-filter motifs to only those whose TFs are in the gene set.

    Matches R's ``motifs_select()``.  The TFs column may contain
    semicolon-separated gene names (one motif -> multiple TFs).
    """
    col = "TFs" if gene_name_type == "symbol" else "EnsemblID"
    hvg_set = set(hvg_names)

    mask = motif_db[col].apply(
        lambda x: any(tf in hvg_set for tf in str(x).split(";"))
    )
    filtered = motif_db[mask].copy()
    logger.info(
        f"  motifs_select: {len(filtered)} / {len(motif_db)} motifs "
        f"have a TF in the gene set"
    )
    return filtered


def _scan_sequence_with_pwm(
    seq: str,
    pwm: np.ndarray,
    threshold: float,
) -> float:
    """Scan a DNA sequence with a PWM and return the best score above threshold.

    Vectorized numpy implementation — scores all positions at once.

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
    Best score above threshold, or -inf if no hit.
    """
    w = pwm.shape[1]
    n = len(seq)

    if n < w:
        return -np.inf

    # Encode sequence as integer array (A=0, C=1, G=2, T=3, other=-1)
    _BASE_LUT = np.full(128, -1, dtype=np.int8)
    _BASE_LUT[ord("A")] = 0
    _BASE_LUT[ord("C")] = 1
    _BASE_LUT[ord("G")] = 2
    _BASE_LUT[ord("T")] = 3
    encoded = _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

    # Build a (n - w + 1, w) matrix of indices for sliding windows
    n_windows = n - w + 1
    window_idx = np.arange(w)[None, :] + np.arange(n_windows)[:, None]
    windows = encoded[window_idx]  # (n_windows, w)

    # Mask windows containing N's (encoded as -1)
    valid = np.all(windows >= 0, axis=1)  # (n_windows,)
    if not valid.any():
        return -np.inf

    # Score all valid windows at once using fancy indexing
    valid_windows = windows[valid]  # (n_valid, w)
    pos_idx = np.arange(w)[None, :]  # (1, w)
    scores = pwm[valid_windows, pos_idx].sum(axis=1)  # (n_valid,)

    best = scores.max()
    return float(best) if best >= threshold else -np.inf


def _estimate_pwm_threshold(pwm: np.ndarray, pvalue: float) -> float:
    """Estimate a PWM score threshold for a given p-value.

    Uses a normal approximation.  This is a fallback when MOODS is not
    available — MOODS computes exact thresholds via dynamic programming.
    """
    mean_score = np.sum(np.mean(pwm, axis=0))
    var_per_pos = np.var(pwm, axis=0)
    sd_score = np.sqrt(np.sum(var_per_pos))

    if sd_score == 0:
        return float(np.sum(np.max(pwm, axis=0)))

    from scipy.stats import norm
    z = norm.ppf(1 - pvalue)
    threshold = mean_score + z * sd_score

    min_score = float(np.sum(np.min(pwm, axis=0)))
    max_score = float(np.sum(np.max(pwm, axis=0)))
    return float(np.clip(threshold, min_score, max_score))


def match_motifs_to_peaks(
    peak_names: list[str] | np.ndarray,
    genome_fasta: Path | str,
    pwm_dict: dict[str, np.ndarray],
    motif_db: pd.DataFrame,
    hvg_names: list[str] | set[str],
    pvalue_cutoff: float = DEFAULT_MOTIF_PVALUE,
    gene_name_type: str = "symbol",
) -> pd.DataFrame:
    """Scan ATAC peaks for TF binding motifs.

    Matches ``Get_motif_peak_pair_df0()`` from the original R code.
    Uses MOODS (exact p-value thresholds) when available, falls back to
    vectorized numpy scanning with a normal-approximation threshold.

    Parameters
    ----------
    peak_names
        ATAC peak names (format: chrX:start-end).
    genome_fasta
        Path to genome FASTA file (mm10 or hg38).
    pwm_dict
        {accession: 4xN numpy array} from ``load_transfac_motifs``.
    motif_db
        Motif-TF mapping DataFrame from ``load_transfac_motifs``.
    hvg_names
        HVG gene names — only motifs whose TF is in this set are scanned
        (matching R's ``motifs_select``).
    pvalue_cutoff
        P-value threshold for motif matching (default: 5^-4 = 0.0016).
    gene_name_type
        'symbol' or 'id'.

    Returns
    -------
    DataFrame with columns: motif_id, peak.
    """
    from pyfaidx import Fasta

    genome_fasta = Path(genome_fasta)

    # Pre-filter motifs to those whose TFs are in the gene set
    filtered_db = _select_motifs(motif_db, hvg_names, gene_name_type)
    motif_accessions = set(filtered_db["Accession"])
    pwms_to_scan = {
        acc: pwm_dict[acc]
        for acc in motif_accessions
        if acc in pwm_dict
    }
    logger.info(
        f"Scanning {len(peak_names)} peaks with {len(pwms_to_scan)} motifs "
        f"(p-value cutoff: {pvalue_cutoff})..."
    )

    if not pwms_to_scan:
        logger.warning("  No motifs to scan!")
        return pd.DataFrame(columns=["motif_id", "peak"])

    # Load genome
    genome = Fasta(str(genome_fasta))

    # Extract peak sequences
    sequences = []
    valid_peaks = []
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
        if seq:
            sequences.append(seq)
            valid_peaks.append(peak)

    logger.info(f"  Extracted {len(sequences)} peak sequences")

    # Try MOODS (exact p-value via DP), fall back to numpy scanner
    try:
        import MOODS.scan
        import MOODS.tools
        use_moods = True
    except ImportError:
        use_moods = False

    results = []
    accessions = sorted(pwms_to_scan.keys())

    if use_moods:
        logger.info("  Using MOODS for exact p-value matching")
        bg = [0.25, 0.25, 0.25, 0.25]

        # Prepare matrices: for each motif, forward + reverse complement
        matrices = []
        matrix_accessions = []  # track which accession each matrix pair belongs to
        for acc in accessions:
            pwm = pwms_to_scan[acc]
            # MOODS expects list-of-lists, rows = A/C/G/T
            mat = [pwm[i].tolist() for i in range(4)]
            mat_rc = [pwm[3 - i][::-1].tolist() for i in range(4)]
            matrices.extend([mat, mat_rc])
            matrix_accessions.extend([acc, acc])

        # Compute exact thresholds
        thresholds = [
            MOODS.tools.threshold_from_p(m, bg, pvalue_cutoff)
            for m in matrices
        ]

        # Scan all sequences
        scanner = MOODS.scan.Scanner(7)
        scanner.set_motifs(matrices, bg, thresholds)

        for i, seq in enumerate(sequences):
            hits = scanner.scan(seq)
            # hits is a list of lists, one per matrix
            for j in range(0, len(hits), 2):  # step by 2 (fwd + rc)
                if hits[j] or hits[j + 1]:
                    acc_idx = j // 2
                    results.append({
                        "motif_id": accessions[acc_idx],
                        "peak": valid_peaks[i],
                    })
            if (i + 1) % 500 == 0:
                logger.info(f"  Scanned {i + 1} / {len(sequences)} peaks...")
    else:
        logger.info("  MOODS not available, using numpy fallback (approximate thresholds)")
        # Precompute thresholds and reverse complements
        motif_data = []
        for acc in accessions:
            pwm = pwms_to_scan[acc]
            threshold = _estimate_pwm_threshold(pwm, pvalue_cutoff)
            pwm_rc = pwm[[3, 2, 1, 0], ::-1]
            motif_data.append((acc, pwm, pwm_rc, threshold))

        for i, seq in enumerate(sequences):
            for acc, pwm_fwd, pwm_rc, threshold in motif_data:
                score_fwd = _scan_sequence_with_pwm(seq, pwm_fwd, threshold)
                score_rc = _scan_sequence_with_pwm(seq, pwm_rc, threshold)
                if max(score_fwd, score_rc) > -np.inf:
                    results.append({"motif_id": acc, "peak": valid_peaks[i]})
            if (i + 1) % 500 == 0:
                logger.info(f"  Scanned {i + 1} / {len(sequences)} peaks...")

    logger.info(f"  Scanned {len(sequences)} peaks total")

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.drop_duplicates()

    logger.info(
        f"  Found {len(result_df)} motif-peak matches "
        f"({result_df['motif_id'].nunique() if len(result_df) > 0 else 0} unique motifs, "
        f"{result_df['peak'].nunique() if len(result_df) > 0 else 0} unique peaks)"
    )

    return result_df


# =========================================================================
#  3d. TF-peak-gene triplet assembly
# =========================================================================


def assemble_triplets(
    correlated_pairs: pd.DataFrame,
    motif_matches: pd.DataFrame,
    motif_db: pd.DataFrame,
    hvg_names: list[str] | np.ndarray,
    gene_name_type: str = "symbol",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine correlated gene-peak pairs with motif matches into triplets.

    Matches ``peak_gene_TF_match()`` from the original R code.
    Uses the TRANSFAC motif_database to map motif accessions -> TF gene names,
    exactly as R's ``get_TF_motif_pair()`` + ``merge()`` does.

    Parameters
    ----------
    correlated_pairs
        DataFrame with columns: gene, peak, spearman_r.
    motif_matches
        DataFrame with columns: motif_id, peak.
    motif_db
        TRANSFAC motif database with columns Accession, TFs, EnsemblID.
    hvg_names
        List of all HVG names (to determine which TFs are in our gene set).
    gene_name_type
        'symbol' uses TFs column, 'id' uses EnsemblID.

    Returns
    -------
    Tuple of:
        - triplets: DataFrame with columns: TF, peak, target_gene, spearman_r
        - gene_labels: DataFrame with columns: gene, type (TF/target),
          associated_peaks (comma-separated), associated_TFs (comma-separated)
    """
    logger.info("Assembling TF-peak-gene triplets...")

    hvg_set = set(hvg_names)
    tf_col = "TFs" if gene_name_type == "symbol" else "EnsemblID"

    # Build motif_id -> TF mapping (matching R's get_TF_motif_pair).
    # Each motif can map to multiple TFs (semicolon-separated).
    tf_motif_pairs = []
    for _, row in motif_db.iterrows():
        accession = row["Accession"]
        for tf in str(row[tf_col]).split(";"):
            tf = tf.strip()
            if tf and tf in hvg_set:
                tf_motif_pairs.append({"motif_id": accession, "TF": tf})
    tf_motif_df = pd.DataFrame(tf_motif_pairs).drop_duplicates()
    logger.info(
        f"  TF-motif pairs in HVG set: {len(tf_motif_df)} "
        f"({tf_motif_df['TF'].nunique()} unique TFs)"
    )

    # Join: motif_matches (motif_id, peak) + tf_motif_df (motif_id, TF)
    # -> (TF, peak)
    tf_peak = motif_matches.merge(tf_motif_df, on="motif_id", how="inner")
    tf_peak = tf_peak[["TF", "peak"]].drop_duplicates()

    # Join with correlated_pairs (gene, peak) -> triplets (TF, peak, target_gene)
    triplets = correlated_pairs.merge(
        tf_peak,
        on="peak",
        how="inner",
    )
    triplets = triplets.rename(columns={"gene": "target_gene"})

    # Remove self-regulation (TF == target)
    triplets = triplets[triplets["TF"] != triplets["target_gene"]].copy()

    logger.info(
        f"  Assembled {len(triplets)} triplets "
        f"({triplets['TF'].nunique()} TFs -> {triplets['target_gene'].nunique()} targets "
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
    pwm_dict: dict[str, np.ndarray] | None = None,
    motif_db: pd.DataFrame | None = None,
    upstream_bp: int = DEFAULT_UPSTREAM_BP,
    downstream_bp: int = DEFAULT_DOWNSTREAM_BP,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    motif_pvalue: float = DEFAULT_MOTIF_PVALUE,
    gene_name_type: str = "symbol",
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
        Path to genome FASTA (required for motif matching).
    pwm_dict
        TRANSFAC PWMs from ``load_transfac_motifs`` (required for motif matching).
    motif_db
        TRANSFAC motif database from ``load_transfac_motifs`` (required for motif matching).
    upstream_bp, downstream_bp
        TSS window for peak-gene overlap.
    corr_threshold
        Minimum |Spearman r| for gene-peak pairs.
    motif_pvalue
        P-value cutoff for motif matching.
    gene_name_type
        'symbol' or 'id' — which TRANSFAC column to use for TF names.
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

    # 3c: Motif matching
    # Only scan peaks that survived correlation filtering (matching R's
    # reduce(peak_gene_overlap_GR2) which only scans correlated peaks).
    if genome_fasta is not None and pwm_dict is not None and motif_db is not None:
        correlated_peak_names = sorted(correlated_pairs["peak"].unique())
        logger.info(
            f"  Motif scanning restricted to {len(correlated_peak_names)} "
            f"correlated peaks (of {len(peak_names)} total)"
        )
        motif_matches = match_motifs_to_peaks(
            correlated_peak_names,
            genome_fasta,
            pwm_dict,
            motif_db,
            hvg_names,
            pvalue_cutoff=motif_pvalue,
            gene_name_type=gene_name_type,
        )
    else:
        logger.warning(
            "  Skipping motif matching (genome FASTA or TRANSFAC files not provided). "
            "Triplet assembly will be incomplete."
        )
        motif_matches = pd.DataFrame(columns=["motif_id", "peak"])

    # 3d: Triplet assembly
    triplets, gene_labels = assemble_triplets(
        correlated_pairs,
        motif_matches,
        motif_db if motif_db is not None else pd.DataFrame(columns=["Accession", "TFs", "EnsemblID"]),
        hvg_names,
        gene_name_type=gene_name_type,
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


# =========================================================================
#  Main
# =========================================================================


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path("data/processed")
    ref_dir = Path("data/reference")
    paper_ref = Path("data/paper/reference")
    seaad_dir = data_dir / "seaad"

    # Load TRANSFAC PWMs (shared across datasets)
    pwm_rds = paper_ref / "all_motif_pwm.rds"
    if not pwm_rds.exists():
        logger.error(f"TRANSFAC PWM file not found: {pwm_rds}")
        sys.exit(1)

    logger.info("Loading TRANSFAC motif data...")
    # We load PWMs once; motif_db is species-specific (loaded per dataset).

    # --- PBMC (paired, human hg38) ---
    pbmc_rna_path = data_dir / "pbmc_rna_sub.h5ad"
    pbmc_atac_path = data_dir / "pbmc_atac_sub.h5ad"

    if pbmc_rna_path.exists() and pbmc_atac_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("PBMC Phase 3")
        logger.info("=" * 60)

        pwm_dict, motif_db_hs = load_transfac_motifs(
            pwm_rds, paper_ref / "Tranfac201803_Hs_MotifTFsFinal",
        )
        gene_ann = load_gene_annotations(paper_ref / "gencode.v38.annotation.gtf")
        pbmc_rna = ad.read_h5ad(pbmc_rna_path)
        pbmc_atac = ad.read_h5ad(pbmc_atac_path)

        run_phase3(
            pbmc_rna, pbmc_atac, gene_ann,
            genome_fasta=ref_dir / "hg38.fa",
            pwm_dict=pwm_dict, motif_db=motif_db_hs,
            output_dir=data_dir, prefix="pbmc",
        )
        del pbmc_rna, pbmc_atac

    # --- Retinal (unpaired, mouse mm10) ---
    ret_rna_path = data_dir / "retinal_rna_sub.h5ad"
    ret_atac_path = data_dir / "retinal_atac_sub.h5ad"

    if ret_rna_path.exists() and ret_atac_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("Retinal Phase 3")
        logger.info("=" * 60)

        pwm_dict_mm, motif_db_mm = load_transfac_motifs(
            pwm_rds, paper_ref / "Tranfac201803_Mm_MotifTFsFinal.txt",
        )
        gene_ann_mm = load_gene_annotations(paper_ref / "mouse.genes.gtf")
        ret_rna = ad.read_h5ad(ret_rna_path)
        ret_atac = ad.read_h5ad(ret_atac_path)

        run_phase3(
            ret_rna, ret_atac, gene_ann_mm,
            genome_fasta=ref_dir / "mm10.fa",
            pwm_dict=pwm_dict_mm, motif_db=motif_db_mm,
            output_dir=data_dir, prefix="retinal",
        )
        del ret_rna, ret_atac

    # --- SEA-AD (paired + unpaired, human hg38) ---
    #
    # Multi-run pattern: each subsample written by a student lives at
    #   data/processed/seaad/seaad_{paired,unpaired}_{rna,atac}_sub{seed}.h5ad
    # where {seed} is the random seed used during subsampling (e.g. "42").
    # Files written with the bare "_sub.h5ad" suffix (no seed) are also picked
    # up, for backward compatibility with the paper's single-subsample design.
    #
    # SEA-AD branches use the Ensembl 98 GTF (matches Phase 1 unpaired and
    # annotates all 500 SEA-AD HVGs — GENCODE v38 silently drops ~26% of them
    # because of Ensembl-style novel-transcript IDs like AL358075.2).
    seaad_hs_loaded = False
    for branch in ("paired", "unpaired"):
        rna_files = sorted(seaad_dir.glob(f"seaad_{branch}_rna_sub*.h5ad"))
        if not rna_files:
            continue

        if not seaad_hs_loaded:
            logger.info("\n" + "=" * 60)
            logger.info("SEA-AD Phase 3 (Ensembl 98)")
            logger.info("=" * 60)
            pwm_dict, motif_db_hs = load_transfac_motifs(
                pwm_rds, paper_ref / "Tranfac201803_Hs_MotifTFsFinal",
            )
            gene_ann_seaad = load_gene_annotations(
                ref_dir / "hg38.ensembl98.gtf.gz",
            )
            seaad_hs_loaded = True

        for rna_path in rna_files:
            # Extract suffix after "_sub" (the seed, possibly empty)
            suffix = rna_path.stem.split("_sub", 1)[1]
            atac_path = seaad_dir / f"seaad_{branch}_atac_sub{suffix}.h5ad"
            if not atac_path.exists():
                logger.warning(
                    f"  Skipping {rna_path.name}: no matching ATAC at "
                    f"{atac_path.name}"
                )
                continue

            run_prefix = (
                f"seaad_{branch}_sub{suffix}" if suffix else f"seaad_{branch}"
            )
            logger.info(
                f"\n  --- {run_prefix} "
                f"(rna={rna_path.name}, atac={atac_path.name}) ---"
            )

            rna_a = ad.read_h5ad(rna_path)
            atac_a = ad.read_h5ad(atac_path)

            run_phase3(
                rna_a, atac_a, gene_ann_seaad,
                genome_fasta=ref_dir / "hg38.fa",
                pwm_dict=pwm_dict, motif_db=motif_db_hs,
                output_dir=seaad_dir, prefix=run_prefix,
            )
            del rna_a, atac_a

    logger.info("\nDone.")
