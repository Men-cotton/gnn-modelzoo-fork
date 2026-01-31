import os
import sys
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr, BaseData
from torch_geometric.datasets import Reddit, Planetoid
import os.path as osp
from cerebras.modelzoo.config.types import resolve_path

try:
    from torch_geometric.distributed import (
        LocalFeatureStore,
        LocalGraphStore,
    )
    from torch_geometric.distributed.dist_context import DistContext
    try:
        from torch_geometric.distributed import DistNeighborLoader
    except ImportError:
        from torch_geometric.loader import DistNeighborLoader
    HAS_DIST = True
except ImportError:
    HAS_DIST = False
    LocalFeatureStore = object
    LocalGraphStore = object
    DistContext = None
    DistNeighborLoader = None

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


def _require_sampler_seed(loader_cfg, context: str) -> int:
    seed = loader_cfg.get("sampler_seed")
    if seed is None:
        raise ValueError(
            f"sampler_seed must be set for {context} to ensure reproducibility."
        )
    return seed


def _build_generator(loader_cfg, context: str):
    seed = _require_sampler_seed(loader_cfg, context)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _mask_to_index(mask):
    if mask is None:
        return None
    if mask.dim() > 1:
        mask = mask[:, 0]
    return mask.bool().nonzero(as_tuple=False).view(-1)


def _resolve_data_dir(data_dir):
    return os.path.abspath(resolve_path(data_dir))


def _normalize_ogb_split_idx(split_idx):
    if not split_idx:
        return split_idx
    sample = next(iter(split_idx.values()))
    if isinstance(sample, dict):
        if "paper" not in sample:
            raise KeyError(
                "OGB MAG split indices missing 'paper' entry. "
                f"Available keys: {', '.join(sorted(sample.keys()))}"
            )
        return {k: v["paper"] for k, v in split_idx.items()}
    return split_idx


def _masks_to_split_idx(data):
    split_idx = {}
    for key, attr in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        mask = getattr(data, attr, None)
        if mask is not None:
            split_idx[key] = _mask_to_index(mask)
    return split_idx


def check_pyg_lib():
    try:
        import pyg_lib  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "This script requires 'pyg_lib' to be installed for efficient neighbor sampling. "
            "Please install it via: pip install pyg_lib -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"
        )


def load_dist_partition(partition_dir, partition_idx):
    if not HAS_DIST:
        raise RuntimeError("torch_geometric.distributed is required for partition loading")
    
    print(f"[loader] Loading partition {partition_idx} from {partition_dir}")
    feat_store = LocalFeatureStore.from_partition(partition_dir, partition_idx)
    graph_store = LocalGraphStore.from_partition(partition_dir, partition_idx)
    
    # Infer dataset name and root from partition_dir
    # partition_dir is .../{num_parts}-parts/{dataset_name}-partitions
    part_path = osp.normpath(partition_dir)
    parts_root = osp.dirname(part_path) # .../{num_parts}-parts
    dataset_partitions_name = osp.basename(part_path) # {dataset_name}-partitions
    dataset_name = dataset_partitions_name.replace("-partitions", "")

    # Load node map to filter indices by ownership
    # node_map.pt is inside the partition_dir (e.g. .../ogbn-arxiv-partitions/node_map.pt)
    node_map_path = osp.join(partition_dir, "node_map.pt")
    if osp.exists(node_map_path):
        print(f"[loader] Loading node map from {node_map_path}")
        node_map = torch.load(node_map_path)
    else:
        print(f"[warn] node_map.pt not found at {node_map_path}. Indices might not be filtered by ownership.", file=sys.stderr)
        node_map = None

    # Load split indices from side-car directories
    # Structure: .../{num_parts}-parts/{dataset_name}-{split}-partitions/partition{idx}.pt
    split_idx = {}
    for split in ["train", "val", "test", "valid"]:
        # Handle 'valid' vs 'val' naming in filesystem
        fs_split = "val" if split == "valid" else split
        
        split_dir = osp.join(parts_root, f"{dataset_name}-{fs_split}-partitions")
        split_file = osp.join(split_dir, f"partition{partition_idx}.pt")
        
        if osp.exists(split_file):
            print(f"[loader] Loading {split} indices from {split_file}")
            idx = torch.load(split_file)

            # Filter by ownership if node_map is available
            if node_map is not None:
                mask = (node_map[idx] == partition_idx)
                original_size = idx.numel()
                idx = idx[mask]
                filtered_size = idx.numel()
                if original_size != filtered_size:
                    print(f"[loader] Filtered {split} indices by ownership: {original_size} -> {filtered_size}")

            split_idx[split] = idx
    
    return (feat_store, graph_store), split_idx


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
    data_dir = _resolve_data_dir(profile["data_dir"])

    check_dataset_exists(data_dir, dataset_name)

    lower = dataset_name.lower()

    if lower.startswith("ogbn-"):
        # OGB datasets are stored in a subdirectory named after the dataset (e.g. ogbn-arxiv)
        # PygNodePropPredDataset appends the underscored name (e.g. ogbn_arxiv) to root.
        # So we need to pass root=data_dir/dataset_name
        dataset = NoDownloadPygNodePropPredDataset(name=dataset_name, root=os.path.join(data_dir, dataset_name))
        data = dataset[0]
        split_idx = _normalize_ogb_split_idx(dataset.get_idx_split())

        if lower == "ogbn-mag":
            if not hasattr(data, "x_dict") or not hasattr(data, "edge_index_dict"):
                raise RuntimeError(
                    "ogbn-mag expected hetero dict fields (x_dict/edge_index_dict). "
                    "Please ensure the dataset was processed with ogb's hetero pipeline."
                )

            if "paper" not in data.x_dict:
                raise RuntimeError(
                    "ogbn-mag expects a 'paper' node type in x_dict."
                )

            edge_index = None
            for edge_type, edge_idx in data.edge_index_dict.items():
                if (
                    isinstance(edge_type, (tuple, list))
                    and len(edge_type) == 3
                    and edge_type[0] == "paper"
                    and edge_type[2] == "paper"
                ):
                    edge_index = edge_idx
                    break

            if edge_index is None:
                raise RuntimeError(
                    "ogbn-mag requires a paper-to-paper edge type. "
                    f"Available edge types: {list(data.edge_index_dict.keys())}"
                )

            paper_x = data.x_dict["paper"]
            if hasattr(data, "y_dict") and "paper" in data.y_dict:
                paper_y = data.y_dict["paper"]
            else:
                paper_y = getattr(data, "y", None)
            if paper_y is None:
                raise RuntimeError(
                    "ogbn-mag labels not found. Expected y_dict['paper']."
                )

            num_nodes = None
            if hasattr(data, "num_nodes_dict"):
                num_nodes = data.num_nodes_dict.get("paper")
            if num_nodes is None:
                num_nodes = paper_x.size(0)

            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
            data = Data(
                x=paper_x,
                edge_index=edge_index,
                y=paper_y.view(-1),
            )
            data.num_nodes = num_nodes
        else:
            # CSZoo reference uses undirected graph
            if data.edge_index is not None:
                data.edge_index = to_undirected(
                    data.edge_index, num_nodes=data.num_nodes
                )
            data.y = data.y.view(-1)

        split_idx = {
            k: v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.long)
            for k, v in split_idx.items()
        }
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
    if key in split_idx:
        return split_idx[key]
    raise KeyError(
        f"Split '{split_name}' not available in dataset. "
        f"Available splits: {', '.join(sorted(split_idx.keys()))}"
    )


def make_loaders(data, split_idx, cfg, rank=0, world_size=1):
    # 参照するブロック
    fit = cfg["trainer"]["fit"]
    train_c = fit["train_dataloader"]
    val_c = cfg["trainer"]["validate"]["val_dataloader"]

    train_profile = _resolve_dataset_profile(train_c)
    val_profile = _resolve_dataset_profile(val_c)

    _require_sampler_seed(train_c, "train_dataloader")
    _require_sampler_seed(val_c, "val_dataloader")

    is_dist = (
        isinstance(data, tuple)
        and len(data) == 2
        and HAS_DIST
        and isinstance(data[0], LocalFeatureStore)
    )

    # Shard training indices for DDP
    train_input_nodes = _resolve_split(split_idx, train_c.get("split", "train"))
    
    if world_size > 1 and not is_dist:
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
    
    # Common dataloader kwargs
    def _get_loader_kwargs(loader_cfg, loader_cls):
        num_workers = loader_cfg.get("num_workers", 0)
        if num_workers <= 0:
            raise ValueError(
                f"num_workers must be > 0 for HPC performance. Got {num_workers}."
            )

        kwargs = {
            "batch_size": loader_cfg["batch_size"],
            "num_workers": num_workers,
            "persistent_workers": loader_cfg.get("persistent_workers", True),
            "pin_memory": loader_cfg.get("pin_memory", True),
        }

        # Validate persistent_workers support
        # NeighborLoader inherits from torch.utils.data.DataLoader or compatible
        # We can check signature or just rely on PyG/Torch usually supporting it if > 0 workers
        # However, user requested strict check.
        # torch.utils.data.DataLoader has supported persistent_workers for a long time (since 1.7+).
        # We assume standard environment. But to be safe/strict as requested:
        import inspect
        sig = inspect.signature(loader_cls.__init__)
        
        if "persistent_workers" not in sig.parameters and "kwargs" not in sig.parameters:
             raise RuntimeError(f"persistent_workers not supported by {loader_cls.__name__}")
        
        if "pin_memory" not in sig.parameters and "kwargs" not in sig.parameters:
             raise RuntimeError(f"pin_memory not supported by {loader_cls.__name__}")

        # prefetch_factor validation
        prefetch_factor = loader_cfg.get("prefetch_factor", 10)
        if "prefetch_factor" in sig.parameters or "kwargs" in sig.parameters:
             kwargs["prefetch_factor"] = prefetch_factor
        else:
             raise RuntimeError(f"prefetch_factor not supported by {loader_cls.__name__}")
             
        # Check if installed PyG version's loader actually accepts these.
        # PyG NeighborLoader passes kwargs to torch DataLoader. 
        # So we really need to check if torch DataLoader supports them, which it does in modern versions.
        
        return kwargs

    if is_dist:
        # Distributed mode with partitions
        print(f"[loader] Using DistNeighborLoader with partitions")
        feature_store, graph_store = data
        
        # DistNeighborLoader requires distributed context info
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        current_ctx = DistContext(
            rank=rank,
            global_rank=rank,
            world_size=world_size,
            global_world_size=world_size,
            group_name="worker",
        )
        
        # We use DistNeighborLoader
        loader_kwargs = _get_loader_kwargs(train_c, DistNeighborLoader)

        train_loader = DistNeighborLoader(
            data=(feature_store, graph_store),
            master_addr=master_addr,
            master_port=22345,
            current_ctx=current_ctx,
            input_nodes=train_input_nodes,
            num_neighbors=_get_fanouts(train_c, train_profile),
            shuffle=train_c["shuffle"],
            drop_last=train_c["drop_last_batch"],
            **loader_kwargs,
        )
        
        try:
            val_nodes = _resolve_split(split_idx, val_c.get("split", "val"))
        except KeyError:
            val_nodes = None

        # DistNeighborLoader requires input_nodes to be provided
        if val_nodes is None:
             raise RuntimeError(
                 "Validation nodes not found in loaded partition data. "
                 "Cannot proceed with validation. Please ensure validation split exists."
             )
        
        val_loader_kwargs = _get_loader_kwargs(val_c, DistNeighborLoader)
        val_loader = DistNeighborLoader(
            data=(feature_store, graph_store),
            master_addr=master_addr,
            master_port=22345,
            current_ctx=current_ctx,
            input_nodes=val_nodes,
            num_neighbors=_get_fanouts(val_c, val_profile),
            shuffle=False,
            drop_last=val_c["drop_last_batch"],
            **val_loader_kwargs,
        )
        return train_loader, val_loader

    train_kwargs = _get_loader_kwargs(train_c, NeighborLoader)
    train_loader = NeighborLoader(
        data,
        input_nodes=train_input_nodes,
        num_neighbors=_get_fanouts(train_c, train_profile),
        shuffle=train_c["shuffle"],
        drop_last=train_c["drop_last_batch"],
        generator=_build_generator(train_c, "train_dataloader"),
        **train_kwargs,
    )

    val_kwargs = _get_loader_kwargs(val_c, NeighborLoader)
    val_loader = NeighborLoader(
        data,
        input_nodes=_resolve_split(split_idx, val_c.get("split", "val")),
        num_neighbors=_get_fanouts(val_c, val_profile),
        shuffle=val_c["shuffle"],
        drop_last=val_c["drop_last_batch"],
        generator=_build_generator(val_c, "val_dataloader"),
        **val_kwargs,
    )
    return train_loader, val_loader
