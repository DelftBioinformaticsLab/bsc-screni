"""Shared utilities for data processing."""

import gzip
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
