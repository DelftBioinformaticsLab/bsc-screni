"""Check the cell2map environment: GPU visible, Cell2Map and Tangram import."""

import scanpy
import tangram
import torch
import torch_geometric

from screni.cell2map import mapping as mp

mp._cell2map_modules()
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU: {gpu}")
print(f"torch_geometric {torch_geometric.__version__}, tangram {getattr(tangram, '__version__', '?')}, "
      f"scanpy {scanpy.__version__}")
print("Cell2Map import OK")
