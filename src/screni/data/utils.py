"""Shared utilities for data processing."""

import gzip
import logging
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def parse_peak_name(peak: str) -> tuple[str, int, int]:
    """Parse a peak name like 'chr1:1000-2000' or 'chr1-1000-2000'.

    Returns (chrom, start, end).
    """
    # Try chr1:1000-2000 format first
    match = re.match(r"^(chr\w+):(\d+)-(\d+)$", peak)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    # Try chr1-1000-2000 format
    match = re.match(r"^(chr\w+)-(\d+)-(\d+)$", peak)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    raise ValueError(f"Cannot parse peak name: {peak}")


def standardize_peak_name(peak: str) -> str:
    """Convert any peak name format to 'chrX:start-end'."""
    chrom, start, end = parse_peak_name(peak)
    return f"{chrom}:{start}-{end}"


def _parse_gtf_attr(attr_string: str, key: str) -> str | None:
    """Extract a value from a GTF attribute string."""
    for attr in attr_string.split(";"):
        attr = attr.strip()
        if attr.startswith(f'{key} "'):
            return attr.split('"')[1]
    return None


def load_gene_annotations(gtf_path: Path | str) -> pd.DataFrame:
    """Load gene annotations from an Ensembl GTF file.

    Extracts gene-level records with TSS coordinates.

    Returns
    -------
    DataFrame with columns: Chromosome, Start, End, Strand, gene_name, gene_id.
    """
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


def peaks_to_dataframe(peak_names: list[str] | np.ndarray) -> pd.DataFrame:
    """Convert peak names to a DataFrame with Chromosome, Start, End columns.

    Suitable for creating a PyRanges object.
    """
    records = []
    for p in peak_names:
        try:
            chrom, start, end = parse_peak_name(p)
            records.append({"Chromosome": chrom, "Start": start, "End": end, "Name": p})
        except ValueError:
            continue
    return pd.DataFrame(records)


def compute_gene_activity(
    atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    upstream_bp: int = 2000,
) -> ad.AnnData:
    """Compute gene activity scores from ATAC peak counts.

    For each gene, sums the accessibility of peaks overlapping the gene body
    extended by ``upstream_bp`` upstream of TSS. Python equivalent of Signac's
    ``GeneActivity()``.

    Uses sparse matrix multiplication for efficiency:
    ``activity = atac.X @ overlap_matrix``.
    """
    logger.info(
        f"Computing gene activity for {atac.n_obs} cells, "
        f"{len(gene_annotations)} genes..."
    )

    gene_ann = gene_annotations.copy().reset_index(drop=True)
    gene_starts = gene_ann["Start"].copy()
    gene_ends = gene_ann["End"].copy()

    plus_mask = gene_ann["Strand"] == "+"
    minus_mask = gene_ann["Strand"] == "-"

    extended_starts = gene_starts.copy()
    extended_ends = gene_ends.copy()
    extended_starts[plus_mask] = np.maximum(0, gene_starts[plus_mask] - upstream_bp)
    extended_ends[minus_mask] = gene_ends[minus_mask] + upstream_bp

    gene_ann = gene_ann.assign(
        ext_start=extended_starts.astype(int),
        ext_end=extended_ends.astype(int),
    )

    peak_df = peaks_to_dataframe(atac.var_names)
    if len(peak_df) == 0:
        raise ValueError("No valid peaks parsed from var_names")

    peak_to_idx = {p: i for i, p in enumerate(atac.var_names)}

    unique_genes = gene_ann["gene_name"].unique()
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    n_peaks = atac.n_vars
    n_genes = len(unique_genes)

    logger.info(f"  Building overlap matrix ({n_peaks} peaks x {n_genes} genes)...")
    overlap_rows = []
    overlap_cols = []

    for chrom in gene_ann["Chromosome"].unique():
        genes_chr = gene_ann[gene_ann["Chromosome"] == chrom]
        peaks_chr = peak_df[peak_df["Chromosome"] == chrom]
        if len(peaks_chr) == 0:
            continue

        p_starts = peaks_chr["Start"].values
        p_ends = peaks_chr["End"].values
        p_names = peaks_chr["Name"].values

        for _, g in genes_chr.iterrows():
            gene_idx = gene_to_idx[g["gene_name"]]
            mask = (p_starts < g["ext_end"]) & (p_ends > g["ext_start"])
            for pname in p_names[mask]:
                if pname in peak_to_idx:
                    overlap_rows.append(peak_to_idx[pname])
                    overlap_cols.append(gene_idx)

        logger.info(f"  Overlaps: {chrom} done ({len(overlap_rows)} pairs so far)")

    overlap = sp.csr_matrix(
        (np.ones(len(overlap_rows), dtype=np.float32), (overlap_rows, overlap_cols)),
        shape=(n_peaks, n_genes),
    )
    logger.info(f"  Overlap matrix: {overlap.nnz} nonzero entries")
    logger.info(f"  Computing activity matrix (sparse matmul)...")
    activity = atac.X @ overlap

    result = ad.AnnData(
        X=activity,
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=unique_genes),
    )

    n_nonzero = (np.array(result.X.sum(axis=0)).flatten() > 0).sum()
    logger.info(
        f"  Gene activity: {result.shape}, "
        f"{n_nonzero} genes with non-zero activity"
    )
    return result
