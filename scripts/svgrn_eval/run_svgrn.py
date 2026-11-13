"""Run SVGRN (external/SVGRN, unmodified) on one simulated dataset.

Stage 1 learns the tissue GRN on all cells; stage 2 fine-tunes it per target
cell. The published settings are used (stage 1: 150 epochs, beta 5,
alpha 100; stage 2: W 6, beta 5, alpha 100, 150 epochs). Ground truth is
never passed to training.

Variants (controls):
    published    as published
    shuffle_rna  every gene permuted across cells independently: destroys all
                 gene-gene structure, keeps each gene's distribution
    shuffle_pos  stage 2 with cell positions permuted: tests whether space
                 adds anything (reuses the published stage-1 model)

Writes <out>/<variant>/stage1/{RN_150.csv,stage1.pt}, per-cell stage-2 output
in <out>/<variant>/cells/<cell>/, and <out>/<variant>/preds_stage{1,2}.npz
(sender x receiver matrices per target cell).

Usage:
    python scripts/svgrn_eval/run_svgrn.py --data <dir>/g110_c2k_n01 \
        --targets <out>/targets.txt --out <out> --variant published
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

SVGRN_DIR = Path(__file__).resolve().parents[2] / "external" / "SVGRN"


def _compat() -> None:
    """Let the 2024-era SVGRN code run on torch >= 2.6 without editing it.

    - torch.autograd.Variable no longer accepts ``name=``.
    - torch.load defaults to weights_only=True, but stage 2 loads the whole
      pickled stage-1 module.
    """
    import torch.autograd as ag

    def variable(data, requires_grad=False, name=None, **_):
        return data.requires_grad_(requires_grad) if requires_grad else data

    ag.Variable = variable
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(SVGRN_DIR))


def _opt(**kw) -> SimpleNamespace:
    """Argument namespace with SVGRN's parser defaults and published settings."""
    base = dict(n_epochs=150, task="simulation_allcell_GRN", setting="default", batch_size=64,
                net_file=None, net_path=None, alpha=100.0, beta=5.0, lr=1e-4, lr_step_size=0.99,
                gamma=0.95, n_hidden=128, K=1, K1=1, K2=2, W=6, model_file="",
                target_cell_name="", GPU=torch.cuda.is_available(), device="",
                dropout_mask=False, adaptive_W=False, k_neighbor=10)
    base.update(kw)
    return SimpleNamespace(**base)


def _rn_to_sender_receiver(path: Path) -> np.ndarray:
    # SVGRN saves adj_A as receiver x sender (eval.py transposes it)
    return pd.read_csv(path).to_numpy().T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True, help="One cell name per line")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=["published", "shuffle_rna", "shuffle_pos"], default="published")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150, help="Published: 150. Lower only for smoke tests")
    args = p.parse_args()
    _compat()
    from src.SVGRN_allcell import non_celltype_GRN_model  # noqa: E402
    from src.SVGRN_singlecell import SC_GRN_model  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    targets = args.targets.read_text().split()[: args.max_targets]

    df = pd.read_csv(args.data / c.INPUT_CSV, index_col=0)
    df.columns = df.columns.astype(str)
    genes = [g for g in df.columns if g not in c.NON_GENE_COLS]
    tfs = list(pd.read_csv(args.data / "allcells_GRN.csv", index_col=0).columns.astype(str))
    if args.variant == "shuffle_rna":
        for g in genes:
            df[g] = rng.permutation(df[g].to_numpy())
    data_csv = out / "input.csv"
    df.to_csv(data_csv)
    tf_file = out / "tf_list.txt"
    tf_file.write_text("\n".join(tfs) + "\n")

    # stage 1 (shuffle_pos reuses the published stage-1 model)
    if args.variant == "shuffle_pos":
        stage1_dir = args.out / "published" / "stage1"
        if not (stage1_dir / "stage1.pt").exists():
            raise SystemExit("shuffle_pos needs the published stage-1 model; run --variant published first")
        pos = df[["x", "y"]].to_numpy()
        df[["x", "y"]] = pos[rng.permutation(len(df))]
        df.to_csv(data_csv)
    else:
        stage1_dir = out / "stage1"
        t0 = time.time()
        non_celltype_GRN_model(_opt(n_epochs=args.epochs, data_file=str(data_csv), tf_list=str(tf_file),
                                    save_name=str(stage1_dir))).train_model()
        print(f"stage 1: {time.time() - t0:.0f} s", flush=True)
    if args.variant != "shuffle_pos":  # same stage 1 as published
        s1 = _rn_to_sender_receiver(stage1_dir / f"RN_{args.epochs}.csv")
        c.save_preds(out / "preds_stage1.npz", {t: s1 for t in targets})

    # stage 2, one fine-tune per target cell
    preds, times = {}, []
    for i, cell in enumerate(targets):
        cell_dir = out / "cells" / cell
        rn = cell_dir / f"RN_{args.epochs}.csv"
        if not rn.exists():
            cell_dir.mkdir(parents=True, exist_ok=True)  # SVGRN only creates the last level
            t0 = time.time()
            SC_GRN_model(_opt(n_epochs=args.epochs, data_file=str(data_csv), model_file=str(stage1_dir / "stage1.pt"),
                              target_cell_name=cell, save_name=str(cell_dir))).train_model()
            times.append(time.time() - t0)
            print(f"stage 2 {i + 1}/{len(targets)} {cell}: {times[-1]:.0f} s", flush=True)
        preds[cell] = _rn_to_sender_receiver(rn)
    c.save_preds(out / "preds_stage2.npz", preds)
    if times:
        print(f"stage 2: {len(times)} cells, median {np.median(times):.0f} s per cell", flush=True)


if __name__ == "__main__":
    main()
