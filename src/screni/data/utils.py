"""Shared utilities for data processing."""

import re

import numpy as np
import pandas as pd


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
