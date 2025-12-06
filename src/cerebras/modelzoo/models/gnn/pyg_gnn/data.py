import os
import torch
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr, BaseData
from torch_geometric.datasets import Reddit, Planetoid
import os.path as osp

_SAFE_GLOBALS = [Data, DataTensorAttr, DataEdgeAttr, BaseData, Data]
add_safe_globals(_SAFE_GLOBALS)

def _resolve_dataset_profile(loader_cfg):
    profiles = loader_cfg.get("dataset_profiles")
    if profiles:
        dataset_key = loader_cfg["dataset"]
        return profiles[dataset_key]
    return loader_cfg


def _get_fanouts(loader_cfg, profile):
    fanouts = loader_cfg.get("fanouts", profile.get("fanouts"))
    if fanouts is None:
        raise ValueError("fanouts must be specified either in loader config or dataset profile")
    return fanouts


def _build_generator(loader_cfg):
    seed = loader_cfg.get("sampler_seed")
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _add_split_aliases(split_idx):
    if "val" in split_idx and "valid" not in split_idx:
        split_idx["valid"] = split_idx["val"]
    if "valid" in split_idx and "val" not in split_idx:
        split_idx["val"] = split_idx["valid"]
    return split_idx


def _mask_to_index(mask):
    if mask is None:
        return None
    if mask.dim() > 1:
        mask = mask[:, 0]
    return mask.bool().nonzero(as_tuple=False).view(-1)


def _masks_to_split_idx(data):
    split_idx = {}
    for key, attr in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        mask = getattr(data, attr, None)
        if mask is not None:
            split_idx[key] = _mask_to_index(mask)
    return _add_split_aliases(split_idx)


def check_pyg_lib():
    try:
        import pyg_lib  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "This script requires 'pyg_lib' to be installed for efficient neighbor sampling. "
            "Please install it via: pip install pyg_lib -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"
        )


def check_dataset_exists(data_dir, dataset_name):
    """Checks if the dataset is already processed and available locally."""
    lower = dataset_name.lower()
    if lower == "reddit":
        # Reddit checks for processed/data.pt
        if not osp.exists(osp.join(data_dir, "Reddit", "processed", "data.pt")):
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found at {data_dir}. "
                "Downloading is disabled on this node. Please prepare the data offline."
            )
    elif lower == "pubmed":
        # Planetoid checks for processed/data.pt
        if not osp.exists(osp.join(data_dir, "PubMed", "processed", "data.pt")):
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found at {data_dir}. "
                "Downloading is disabled on this node. Please prepare the data offline."
            )
    elif lower.startswith("ogbn-"):
        # OGB checks for processed/geometric_data_processed.pt
        # The folder name is usually underscored, e.g. ogbn_arxiv
        # And it is nested inside a folder named after the dataset (hyphenated)
        folder_name = dataset_name.replace("-", "_")
        if not osp.exists(osp.join(data_dir, dataset_name, folder_name, "processed", "geometric_data_processed.pt")):
             raise RuntimeError(
                f"Dataset '{dataset_name}' not found at {data_dir}. "
                "Downloading is disabled on this node. Please prepare the data offline."
            )


class NoDownloadReddit(Reddit):
    def download(self):
        raise RuntimeError(f"Download disabled. Please ensure data is at {self.root}")

class NoDownloadPlanetoid(Planetoid):
    def download(self):
        raise RuntimeError(f"Download disabled. Please ensure data is at {self.root}")

class NoDownloadPygNodePropPredDataset(PygNodePropPredDataset):
    def download(self):
        raise RuntimeError(f"Download disabled. Please ensure data is at {self.root}")

def load_dataset(profile):
    dataset_name = profile["dataset_name"]
    data_dir = profile["data_dir"]
    
    # Resolve data_dir relative to this script if it is relative
    # Note: In original script it was relative to __file__. 
    # Here we should probably keep it relative to CWD or pass absolute path.
    # The original logic:
    # if not os.path.isabs(data_dir):
    #     base_dir = os.path.dirname(os.path.abspath(__file__))
    #     data_dir = os.path.normpath(os.path.join(base_dir, data_dir))
    # Since we are moving file deeper, relative path might break if it relies on script location.
    # However, usually data_dir in config is relative to where you run it or absolute.
    # Let's assume the user runs from root or provides absolute path, OR we need to be careful.
    # The original script was in `src/cerebras/modelzoo/models/gnn/`.
    # New script is in `src/cerebras/modelzoo/models/gnn/pyg_gnn/`.
    # If I use `__file__` here, it will be one level deeper.
    # I should probably adjust the base_dir logic if I keep it.
    # But wait, the original script used `__file__` of `graphsage_pyg.py`.
    # If I move this to `data.py`, `__file__` is `.../pyg_gnn/data.py`.
    # So `os.path.dirname` is `.../pyg_gnn`.
    # Original was `.../gnn`.
    # So I should go one level up if I want to maintain exact behavior relative to the python file location?
    # Actually, `graphsage_pyg.py` will still be in `.../gnn`.
    # If I run `graphsage_pyg.py`, and it calls `load_dataset`, and `load_dataset` uses `__file__` of `data.py`, it will be `.../pyg_gnn`.
    # If the config says `data_dir: ./data`, original meant `.../gnn/data`.
    # New `data.py` would mean `.../pyg_gnn/data`.
    # This is a change.
    # I should probably fix this.
    # Let's check where `data` folder is.
    # `ls -R` showed `data` in `src/cerebras/modelzoo/models/gnn/data`.
    # So we want `.../gnn/data`.
    # So if `data.py` is in `.../gnn/pyg_gnn/data.py`, `dirname` is `.../gnn/pyg_gnn`.
    # `dirname(dirname(__file__))` is `.../gnn`.
    
    if not os.path.isabs(data_dir):
        # Adjusting to match original relative path behavior (relative to .../gnn/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.normpath(os.path.join(base_dir, data_dir))
        print(f"[loader] Resolved data_dir to: {data_dir}")

    check_dataset_exists(data_dir, dataset_name)

    lower = dataset_name.lower()

    if lower.startswith("ogbn-"):
        # OGB datasets are stored in a subdirectory named after the dataset (e.g. ogbn-arxiv)
        # PygNodePropPredDataset appends the underscored name (e.g. ogbn_arxiv) to root.
        # So we need to pass root=data_dir/dataset_name
        dataset = NoDownloadPygNodePropPredDataset(name=dataset_name, root=os.path.join(data_dir, dataset_name))
        data = dataset[0]
        data.y = data.y.view(-1)
        split_idx = dataset.get_idx_split()
        split_idx = {
            k: v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.long)
            for k, v in split_idx.items()
        }
        split_idx = _add_split_aliases(split_idx)
        return data, split_idx

    if lower == "reddit":
        # Reddit dataset in PyG treats root as the dataset directory itself (does not append name)
        # But our data structure is data_dir/Reddit
        dataset = NoDownloadReddit(root=os.path.join(data_dir, "Reddit"))
        data = dataset[0]
        split_idx = _masks_to_split_idx(data)
        return data, split_idx

    if lower == "pubmed":
        # Planetoid appends name to root, so passing data_dir is correct
        dataset = NoDownloadPlanetoid(root=data_dir, name="PubMed")
        data = dataset[0]
        split_idx = _masks_to_split_idx(data)
        return data, split_idx

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _resolve_split(split_idx, split_name):
    key = split_name.lower()
    candidates = [key]
    if key == "val":
        candidates.append("valid")
    elif key == "valid":
        candidates.append("val")
    for cand in candidates:
        if cand in split_idx:
            return split_idx[cand]
    raise KeyError(f"Split '{split_name}' not available in dataset")


def make_loaders(data, split_idx, cfg, rank=0, world_size=1):
    # 参照するブロック
    fit = cfg["trainer"]["fit"]
    train_c = fit["train_dataloader"]
    val_c = cfg["trainer"]["validate"]["val_dataloader"]

    train_profile = _resolve_dataset_profile(train_c)
    val_profile = _resolve_dataset_profile(val_c)

    # Shard training indices for DDP
    train_input_nodes = _resolve_split(split_idx, train_c.get("split", "train"))
    
    if world_size > 1:
        num_nodes = train_input_nodes.size(0)
        # Partition logic similar to OFFSET-GNN baseline
        # Try to split as evenly as possible
        base_size = num_nodes // world_size
        extra = num_nodes % world_size
        
        start = rank * base_size + min(rank, extra)
        length = base_size + (1 if rank < extra else 0)
        
        # Slice the tensor
        train_input_nodes = train_input_nodes[start : start + length]
        print(f"[ddp] Rank {rank}/{world_size}: Assigned {train_input_nodes.size(0)}/{num_nodes} training nodes (Indices {start} to {start + length})")
    
    if rank == 0 and world_size == 1:
         print(f"[loader] Single-process: Using full training set ({train_input_nodes.size(0)} nodes)")

    train_loader = NeighborLoader(
        data,
        input_nodes=train_input_nodes,
        num_neighbors=_get_fanouts(train_c, train_profile),
        batch_size=train_c["batch_size"],
        shuffle=train_c["shuffle"],
        drop_last=train_c["drop_last_batch"],
        num_workers=train_c["num_workers"],
        generator=_build_generator(train_c),
        pin_memory=True,
        persistent_workers=train_c["num_workers"] > 0,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=_resolve_split(split_idx, val_c.get("split", "val")),
        num_neighbors=_get_fanouts(val_c, val_profile),
        batch_size=val_c["batch_size"],
        shuffle=val_c["shuffle"],
        drop_last=val_c["drop_last_batch"],
        num_workers=val_c["num_workers"],
        generator=_build_generator(val_c),
        pin_memory=True,
        persistent_workers=val_c["num_workers"] > 0,
    )
    return train_loader, val_loader
