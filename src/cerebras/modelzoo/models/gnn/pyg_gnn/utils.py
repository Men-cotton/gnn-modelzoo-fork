import random
import os
import numpy as np
import torch
import torch.distributed as dist
import sys
from importlib import import_module
from pathlib import Path
from typing import Iterable, Optional, Union
import yaml

class OfflineDatasetNotFound(FileNotFoundError):
    """Raised when offline=True and required dataset files are not present."""
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # optimize for performance (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def ensure_pickle_friendly_load():
    """
    Patches torch.load to default to weights_only=False.
    This safely suppresses the FutureWarning from libraries like OGB that rely on implicit pickle loading.
    """
    original_load = torch.load
    def patched_load(*args, **kwargs):
        # Set default weights_only=False if not specified
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    torch.load = patched_load

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_ddp():
    """
    Initializes DDP if appropriate environment variables are set.
    Returns:
        rank (int): Global rank
        world_size (int): Total number of processes
        device (torch.device): Assigned device
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Initialize the process group
        dist.init_process_group(backend="nccl")
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        print(f"[ddp] Initialized: rank={rank}, world_size={world_size}, device={device}")
        return rank, world_size, device
    else:
        # Fallback for single GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ddp] Single-process mode: device={device}")
        return 0, 1, device

def cleanup_ddp():
    """Destroys the process group if it exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def _canonicalize_ogb_name(name: str) -> str:
    """Return the official OGB dataset identifier (ogbn-*) for the given name."""
    if name.startswith("ogbn-"):
        return name
    if name.startswith("ogbn_"):
        return name.replace("_", "-")
    return name

def _ogb_name_variants(name: str) -> Iterable[str]:
    """Yield plausible directory names (underscore / hyphen) for an ogbn dataset."""
    seen = set()
    candidates = [
        name,
        name.replace("_", "-") if "_" in name else None,
        name.replace("-", "_") if "-" in name else None,
    ]
    for variant in candidates:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        yield variant

def _resolve_ogb_dir(root: str, name: str) -> Path:
    """
    Returns the actual directory for an OGB dataset (preferring existing raw/processed).
    If none exist, returns the first candidate.
    """
    base = Path(root)
    candidates = [base / variant for variant in _ogb_name_variants(name)]
    if not candidates:
        candidates = [base / name]
    for d in candidates:
        if (d / "processed").exists() or (d / "raw").exists():
            return d
    return candidates[0]

def _ensure_ogb_alias(root: Path, canonical_name: str, actual_dir: Path):
    """
    Creates a symlink if the canonical name (e.g. ogbn-arxiv) differs from the actual directory (e.g. ogbn_arxiv).
    """
    alias_dir = root / canonical_name
    if alias_dir == actual_dir:
        return alias_dir
    if alias_dir.exists():
        return alias_dir
    if not actual_dir.exists():
        return alias_dir
    try:
        alias_dir.symlink_to(actual_dir, target_is_directory=True)
    except FileExistsError:
        pass
    except OSError as exc:
        print(f"[warn] could not create OGB alias {alias_dir} -> {actual_dir}: {exc}", file=sys.stderr)
    return alias_dir if alias_dir.exists() else actual_dir

def _import_optional(path: str):
    try:
        module_path, attr = path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, attr)
    except Exception:
        return None

def _register_safe_globals():
    """Register PyG classes as safe globals for torch.load."""
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        return
    symbols = [
        "torch_geometric.data.Data",
        "torch_geometric.data.HeteroData",
        "torch_geometric.data.DataEdgeAttr",
        "torch_geometric.data.data.DataEdgeAttr",
        "torch_geometric.data.DataTensorAttr",
        "torch_geometric.data.data.DataTensorAttr",
    ]
    safe = []
    for path in symbols:
        obj = _import_optional(path)
        if obj is not None:
            safe.append(obj)
    if safe:
        add_safe_globals(safe)
