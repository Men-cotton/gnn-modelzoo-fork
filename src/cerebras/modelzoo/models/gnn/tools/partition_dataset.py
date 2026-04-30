"""
High-level partitioning logic for OGB/PyG datasets.
Encapsulates environment setup, offline checks, and calling the lower-level partition_graph.
"""

import argparse
import os
import builtins
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Union
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.distributed import Partitioner
from torch_geometric.utils import mask_to_index
from ogb.nodeproppred import PygNodePropPredDataset

from cerebras.modelzoo.models.gnn.reference.pyg.utils import (
    _canonicalize_ogb_name,
    _resolve_ogb_dir,
    _ensure_ogb_alias,
    ensure_pickle_friendly_load,
    _register_safe_globals,
    OfflineDatasetNotFound,
    load_cfg,
)
from cerebras.modelzoo.models.gnn.reference.pyg.data import _resolve_dataset_profile

GNN_ROOT = Path(__file__).resolve().parents[1]


def partition_dataset(
    dataset_name: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
    use_sparse_tensor: bool = False,
    dataset_dir: str = None,
):
    if not osp.isabs(root_dir):
        root_dir = str((GNN_ROOT / root_dir).resolve())

    # Prefer explicitly supplied dataset_dir (actual data location). Otherwise try common layouts.
    if dataset_dir is None:
        candidates = [
            osp.join(root_dir, "dataset", dataset_name),
            osp.join(root_dir, dataset_name),
        ]
        dataset_dir = None
        for cand in candidates:
            if osp.exists(osp.join(cand, "processed")) or osp.exists(
                osp.join(cand, "raw")
            ):
                dataset_dir = cand
                break
        if dataset_dir is None:
            dataset_dir = candidates[0]

    dataset = get_dataset(dataset_name, dataset_dir, use_sparse_tensor)
    data = dataset[0]

    if (
        dataset_name in ("ogbn-products", "ogbn-arxiv")
        and getattr(data, "y", None) is not None
    ):
        data.y = data.y.view(-1)

    save_dir = osp.join(root_dir, "partitions", dataset_name, f"{num_parts}-parts")

    partitions_dir = osp.join(save_dir, f"{dataset_name}-partitions")
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()

    print("-- Saving label ...")
    label_dir = osp.join(save_dir, f"{dataset_name}-label")
    os.makedirs(label_dir, exist_ok=True)

    if dataset_name == "ogbn-mag":
        split_data = data["paper"]
        label = split_data.y
    else:
        split_data = data
        if dataset_name == "ogbn-products":
            label = split_data.y.view(-1)
        elif dataset_name == "ogbn-arxiv":
            label = split_data.y.view(-1)
        elif dataset_name == "Reddit":
            label = split_data.y
        else:
            raise ValueError(f"Unsupported dataset for partitioning: {dataset_name}")

    torch.save(label, osp.join(label_dir, "label.pt"))

    split_idx = get_idx_split(dataset, dataset_name, split_data)
    save_partitions(split_idx, dataset_name, num_parts, save_dir)


def get_dataset(name, dataset_dir, use_sparse_tensor=False):
    # Ensure torch.load behaves like legacy (weights_only=False) and allowlist PyG classes.
    # Using the utils version
    ensure_pickle_friendly_load()
    _register_safe_globals()

    transforms = []
    if use_sparse_tensor:
        transforms = [T.ToSparseTensor(remove_edge_index=False)]

    if name == "ogbn-mag":
        transforms = [T.ToUndirected(merge=True)] + transforms
        return OGB_MAG(
            root=dataset_dir,
            preprocess="metapath2vec",
            transform=T.Compose(transforms),
        )
    elif name == "ogbn-products":
        transforms = [T.RemoveDuplicatedEdges()] + transforms
        # If processed/raw already exist directly under dataset_dir, pass its parent as root
        # so that PygNodePropPredDataset does not try to create a nested subdir.
        from pathlib import Path

        dpath = Path(dataset_dir)
        root_for_pyg = (
            dpath.parent
            if (dpath / "processed").exists() or (dpath / "raw").exists()
            else dpath
        )
        return PygNodePropPredDataset(
            "ogbn-products",
            root=str(root_for_pyg),
            transform=T.Compose(transforms),
        )
    elif name == "ogbn-arxiv":
        transforms = [T.RemoveDuplicatedEdges()] + transforms
        from pathlib import Path

        dpath = Path(dataset_dir)
        root_for_pyg = (
            dpath.parent
            if (dpath / "processed").exists() or (dpath / "raw").exists()
            else dpath
        )
        return PygNodePropPredDataset(
            "ogbn-arxiv",
            root=str(root_for_pyg),
            transform=T.Compose(transforms),
        )
    elif name == "Reddit":
        return Reddit(
            root=dataset_dir,
            transform=T.Compose(transforms),
        )


def get_idx_split(dataset, dataset_name, split_data):
    if dataset_name == "ogbn-mag" or dataset_name == "Reddit":
        train_idx = mask_to_index(split_data.train_mask)
        test_idx = mask_to_index(split_data.test_mask)
        val_idx = mask_to_index(split_data.val_mask)
    elif dataset_name == "ogbn-products" or dataset_name == "ogbn-arxiv":
        split_idx = dataset.get_idx_split()
        train_idx = torch.as_tensor(split_idx["train"], dtype=torch.long).view(-1)
        test_idx = torch.as_tensor(split_idx["test"], dtype=torch.long).view(-1)
        val_idx = torch.as_tensor(split_idx["valid"], dtype=torch.long).view(-1)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_partitions(split_idx, dataset_name, num_parts, save_dir):
    # Load node_map to assign indices to correct partition
    # node_map is at save_dir/../{dataset_name}-partitions/node_map.pt
    # save_dir is .../{num_parts}-parts

    parts_dir = osp.join(save_dir, f"{dataset_name}-partitions")
    node_map_path = osp.join(parts_dir, "node_map.pt")

    if not osp.exists(node_map_path):
        print(
            f"[warn] node_map.pt not found at {node_map_path}. Falling back to tensor_split (contiguous)."
        )
        node_map = None
    else:
        print(f"-- Loading node map from {node_map_path}")
        node_map = torch.load(node_map_path)

    for key, idx in split_idx.items():
        print(f"-- Partitioning {key} indices ...")

        part_dir = osp.join(save_dir, f"{dataset_name}-{key}-partitions")
        os.makedirs(part_dir, exist_ok=True)

        if node_map is not None:
            # Mask indices by ownership
            # idx contains global node IDs. node_map[global_id] -> partition_id
            ownership = node_map[idx]
            for i in range(num_parts):
                mask = ownership == i
                chunk = idx[mask]
                torch.save(chunk, osp.join(part_dir, f"partition{i}.pt"))
        else:
            # Fallback
            idx_chunks = torch.tensor_split(idx, num_parts)
            for i, chunk in enumerate(idx_chunks):
                torch.save(chunk, osp.join(part_dir, f"partition{i}.pt"))


def save_link_partitions(split_idx, data, dataset_name, num_parts, save_dir):
    edge_type = data.edge_types[0]

    for key, idx in split_idx.items():
        print(f"-- Partitioning {key} indices ...")
        idx_chunks = torch.tensor_split(idx, num_parts)

        part_dir = osp.join(save_dir, f"{dataset_name}-{key}-partitions")
        os.makedirs(part_dir, exist_ok=True)
        for i, chunk in enumerate(idx_chunks):
            edge_index = data[edge_type].edge_index[:, chunk]
            label = data[edge_type].edge_label[chunk]
            edge_time = data[edge_type].time[chunk]
            partition = {
                "edge_label_index": edge_index,
                "edge_label": label,
                "edge_label_time": edge_time - 1,
            }
            torch.save(partition, osp.join(part_dir, f"partition{i}.pt"))


@contextmanager
def _suppress_ogb_input():
    """Monkeypatch input() to auto-answer 'n' to avoid OGB download prompts."""
    _orig_input = builtins.input

    def _no_input(prompt=""):
        return "n"

    builtins.input = _no_input
    try:
        yield
    finally:
        builtins.input = _orig_input


def find_dataset_dir(dataset_name: str, root: Union[str, Path]) -> Path:
    """
    Locate the dataset directory (containing raw/ or processed/) under root.
    Handles 'dataset/<name>', 'ogb/<name>', and underscore/hyphen variants.
    Returns the Path to the dataset root. Raises FileNotFoundError if not found.
    """
    root_path = Path(root).resolve()
    ogb_name = _canonicalize_ogb_name(dataset_name)

    # Base search paths: root, root/ogb
    bases = [root_path, root_path / "ogb"]

    # Name variants
    variants = [ogb_name]
    if "-" in ogb_name:
        variants.append(ogb_name.replace("-", "_"))
    if "_" in ogb_name:
        variants.append(ogb_name.replace("_", "-"))

    found_dir = None

    def _has_data_dir(path: Path) -> bool:
        return (path / "processed").exists() or (path / "raw").exists()

    def _nested_variant_dir(parent: Path) -> Optional[Path]:
        for v in variants:
            nested = parent / v
            if _has_data_dir(nested):
                return nested
        return None

    # 1. Try standard resolution using dataio logic first (root/<name>)
    for b in bases:
        if not b.exists():
            continue
        try:
            d = _resolve_ogb_dir(str(b), ogb_name)
            if _has_data_dir(d):
                found_dir = d
                break
            nested = _nested_variant_dir(d)
            if nested is not None:
                found_dir = nested
                break
        except Exception:
            pass

    if found_dir:
        # Ensure canonical alias if we found it via dataio resolution
        _ensure_ogb_alias(found_dir.parent, ogb_name, found_dir)
        return found_dir

    # 2. Try explicit 'dataset/' subdir which dataio might miss
    for b in bases:
        for v in variants:
            d = b / "dataset" / v
            if _has_data_dir(d):
                found_dir = d
                break
            nested = _nested_variant_dir(d)
            if nested is not None:
                found_dir = nested
                break
        if found_dir:
            break

    if found_dir:
        return found_dir

    raise FileNotFoundError(
        f"Dataset {dataset_name} not found under {root_path}. "
        "Expected 'raw/' or 'processed/' directories."
    )


def prepare_partition(
    dataset: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
) -> Path:
    """
    Prepare partitions for the given dataset.

    Args:
        dataset: Name of the dataset (e.g. 'ogbn-arxiv')
        root_dir: Root directory containing datasets
        num_parts: Number of partitions
        recursive: Whether to use recursive partitioning

    Returns:
        Path to the partitions directory
    """
    dataset_name = _canonicalize_ogb_name(dataset.replace("_", "-"))
    if not osp.isabs(root_dir):
        root_dir = str((GNN_ROOT / root_dir).resolve())
    output_root = Path(root_dir).resolve()

    # Ensure dataset exists offline
    try:
        dataset_dir = find_dataset_dir(dataset_name, output_root)
    except FileNotFoundError as e:
        raise OfflineDatasetNotFound(str(e))

    # Downloads are explicitly disabled; dataset must already exist on disk.
    os.environ.setdefault("OGB_SKIP_PROMPT", "1")
    os.environ.setdefault("OGB_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("OGB_ALWAYS_LOCAL", "1")

    # Call the actual partitioner
    with _suppress_ogb_input():
        partition_dataset(
            dataset_name,
            str(output_root),
            num_parts,
            recursive,
            dataset_dir=str(dataset_dir),
        )

    return get_partition_path(dataset_name, output_root, num_parts)


def get_partition_path(
    dataset_name: str, root_dir: Union[str, Path], num_parts: int
) -> Path:
    """Return the expected partition directory path."""
    root_path = Path(root_dir)
    if not root_path.is_absolute():
        root_path = GNN_ROOT / root_path
    output_root = root_path.resolve()
    # Ensure dataset_name is canonical if not already
    dataset_name = _canonicalize_ogb_name(dataset_name.replace("_", "-"))

    # Check if partitions exist in root/ogb/partitions first if root/partitions doesn't have them
    std_path = (
        output_root
        / "partitions"
        / dataset_name
        / f"{num_parts}-parts"
        / f"{dataset_name}-partitions"
    )
    ogb_path = (
        output_root
        / "ogb"
        / "partitions"
        / dataset_name
        / f"{num_parts}-parts"
        / f"{dataset_name}-partitions"
    )

    if not std_path.exists() and ogb_path.exists():
        return ogb_path

    return std_path


def _resolve_config_dataset(config_path: str, dataset_override: Optional[str]):
    cfg = load_cfg(config_path)
    train_c = cfg["trainer"]["fit"]["train_dataloader"]
    dataset_profile = _resolve_dataset_profile(train_c)
    dataset_name = dataset_profile.get("dataset_name", train_c.get("dataset"))
    if dataset_override:
        dataset_name = dataset_override
    data_dir = dataset_profile.get("data_dir")
    if data_dir is None:
        print(
            "[warn] data_dir is missing from dataset_profiles; cannot prepare partitions.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not osp.isabs(data_dir):
        data_dir = str((GNN_ROOT / data_dir).resolve())
    return dataset_name, data_dir


def main():
    ap = argparse.ArgumentParser(
        description="Prepare PyG partitions for distributed training."
    )
    ap.add_argument(
        "--config", help="Path to training YAML (uses dataset_profiles for data_dir)."
    )
    ap.add_argument("--dataset", help="Dataset name (overrides config if set).")
    ap.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Number of partitions to generate.",
    )
    ap.add_argument(
        "--recursive", action="store_true", help="Use recursive partitioning."
    )
    args = ap.parse_args()

    if not args.config:
        print(
            "[warn] --config is required to resolve dataset_profiles and data_dir.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dataset_name, data_dir = _resolve_config_dataset(args.config, args.dataset)

    print(
        f"[Partition] Generating {args.num_partitions} partitions for {dataset_name} in {data_dir}"
    )
    path = prepare_partition(
        dataset_name,
        data_dir,
        args.num_partitions,
        recursive=args.recursive,
    )
    print(f"[Partition] Done. Partitions saved to: {path}")


if __name__ == "__main__":
    main()
