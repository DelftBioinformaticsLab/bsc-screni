"""Combining per-cell wScReNI network files into a single in-memory collection.

Matches the original R function ``Combine_wScReNI_scNetworks()``.

The wScReNI inference step (``Infer_wScReNI_scNetworks``) writes one
tab-separated file per cell to ``<network_dir>/wScReNI/`` using the naming
convention::

    {1-based-index}.{cell_name}.network.txt

This module reads those files back and assembles them into a dict keyed
by cell name, which is the Python equivalent of R's named list of matrices.

File format details
-------------------
Files are written by R's ``write.table(..., sep="\\t")`` with default
``row.names=TRUE`` and ``col.names=TRUE``.  The resulting format is a
tab-separated matrix where:

- The header row lists gene names (with a leading empty field for the
  row-name column).
- Each subsequent row starts with the gene name (row label), followed by
  the numeric weight values.

``pandas.read_csv(..., sep="\\t", index_col=0)`` reads this correctly:
the index becomes the row gene names and the column labels are the column
gene names. Because R also executes ``colnames(mat) <- rownames(mat)``
after reading, row and column gene names are always identical.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScReniNetworks(dict):
    """A dict of ``{cell_name: weight_matrix}`` with an attached gene list.

    Behaves exactly like a plain ``dict[str, np.ndarray]`` for item access,
    iteration, and length, while also exposing a ``.gene_names`` attribute
    that records the ordered list of gene names shared by every matrix.

    This mirrors R's named list of matrices, which carries gene labels as
    ``rownames``/``colnames`` on each element.
    """

    def __init__(self, *args, gene_names: list | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gene_names: list[str] = gene_names if gene_names is not None else []


def combine_wscreni_networks(
    cell_names: list[str],
    network_dir: Path | str,
    cell_indices: list[int] | None = None,
) -> "ScReniNetworks":
    """Read per-cell wScReNI network files from disk and combine into a dict.

    Matches ``Combine_wScReNI_scNetworks()`` from the original R code.

    Each cell's network is stored as a square ``(n_genes, n_genes)`` numpy
    array of regulatory weights.  The dict is keyed by cell name and its
    ordering matches ``cell_names`` (or the requested ``cell_indices``
    subset), which makes it compatible with the cell-order assumptions in
    downstream steps such as precision-recall evaluation and degree
    computation.

    Parameters
    ----------
    cell_names
        Ordered list of cell names.  Corresponds to ``colnames(sub.scatac.top)``
        in the R code.  The position of each name (0-based) determines the
        numeric prefix used in the filename (1-based, matching R's indexing).
    network_dir
        Parent directory that contains the ``wScReNI/`` sub-folder with the
        per-cell ``.network.txt`` files.  Corresponds to ``network.path`` in
        the R code.
    cell_indices
        Optional 0-based indices into ``cell_names`` specifying which cells
        to load.  When ``None`` (default) all cells are loaded, matching
        the R default of ``cell.index <- 1:length(cellnames)``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of cell name -> ``(n_genes, n_genes)`` weight matrix.  The
        dict is ordered (Python 3.7+) and follows the same order as the
        requested ``cell_indices`` (or all cells when ``None``).
        Gene names are NOT stored in this dict; they can be recovered from
        the ``gene_names`` attribute of the returned dict (see below) or
        from the RNA AnnData used during inference.

    Raises
    ------
    FileNotFoundError
        If ``<network_dir>/wScReNI/`` does not exist.

    Notes
    -----
    Files that are missing for individual cells are skipped with a warning
    rather than raising an exception.  This allows partial loading when only
    a subset of cells has been processed so far.

    The returned dict has an extra attribute ``gene_names`` (a list of str)
    populated from the first successfully loaded file.  Access it as::

        networks = combine_wscreni_networks(cell_names, path)
        genes = networks.gene_names  # list[str], length n_genes

    Examples
    --------
    >>> networks = combine_wscreni_networks(
    ...     cell_names=rna_sub.obs_names.tolist(),
    ...     network_dir=Path("results/NetworksC100"),
    ... )
    >>> # Access a single cell's weight matrix
    >>> cell_matrix = networks["cell_barcode_001"]  # shape (n_genes, n_genes)
    """
    network_dir = Path(network_dir)
    wscreni_dir = network_dir / "wScReNI"

    if not wscreni_dir.exists():
        raise FileNotFoundError(
            f"wScReNI network directory not found: {wscreni_dir}\n"
            f"Run Infer_wScReNI_scNetworks first to generate per-cell files."
        )

    if cell_indices is None:
        cell_indices = list(range(len(cell_names)))

    n_requested = len(cell_indices)
    logger.info(
        f"Loading {n_requested} wScReNI network(s) from {wscreni_dir} ..."
    )

    networks = ScReniNetworks()
    gene_names: list[str] | None = None
    n_missing = 0

    for idx in cell_indices:
        cell_name = cell_names[idx]
        # R uses 1-based indexing for the numeric file prefix.
        file_num = idx + 1
        filename = f"{file_num}.{cell_name}.network.txt"
        filepath = wscreni_dir / filename

        if not filepath.exists():
            logger.warning(
                f"  [{file_num}/{n_requested}] Missing: {filename} — skipping"
            )
            n_missing += 1
            continue

        # R writes with write.table(sep="\t", row.names=TRUE, col.names=TRUE).
        # The resulting file has an empty leading field on the header line so
        # pandas' index_col=0 recovers row gene names correctly.
        df = pd.read_csv(filepath, sep="\t", index_col=0)

        # Ensure column labels match row labels (mirrors R's
        # ``colnames(mat) <- rownames(mat)``).  In practice they are already
        # identical when the file was written by Infer_wScReNI_scNetworks, but
        # this makes loading robust to any column-label drift.
        df.columns = df.index

        matrix = df.to_numpy(dtype=np.float64)

        # Capture gene names from the first file; verify consistency for the
        # rest so callers can trust that all matrices share the same axes.
        if gene_names is None:
            gene_names = df.index.tolist()
        elif df.index.tolist() != gene_names:
            logger.warning(
                f"  Gene name mismatch in {filename}: "
                f"expected {len(gene_names)} genes matching the first file, "
                f"got {len(df.index)}. This cell's matrix may be misaligned."
            )

        networks[cell_name] = matrix
        logger.info(f"  [{file_num}] Loaded: {cell_name} ({matrix.shape})")

    n_loaded = len(networks)
    if n_missing > 0:
        logger.warning(
            f"Loaded {n_loaded}/{n_requested} networks "
            f"({n_missing} file(s) missing)."
        )
    else:
        logger.info(f"Loaded {n_loaded}/{n_requested} networks successfully.")

    networks.gene_names = gene_names if gene_names is not None else []

    return networks
