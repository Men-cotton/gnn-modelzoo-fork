from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:  # pragma: no cover - optional dependency
    PygNodePropPredDataset = None  # type: ignore[assignment]

DATASET_NAME = "ogbn-papers100M"
DIR_NAME = "ogbn_papers100M"
SPLIT_NAME = "time"

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "datasets"


def _resolve_root(root: str | None) -> Path:
    if root is None:
        root = str(DEFAULT_ROOT)
    resolved = os.path.abspath(os.path.expanduser(root))
    return Path(resolved)


def _select_dataset_dir(root: Path) -> Path:
    primary = root / DIR_NAME
    pyg_variant = root / f"{DIR_NAME}_pyg"
    if pyg_variant.exists():
        return pyg_variant
    return primary


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _check_raw(dataset_dir: Path) -> None:
    raw_dir = dataset_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    data_npz = raw_dir / "data.npz"
    label_npz = raw_dir / "node-label.npz"
    _require_file(data_npz, "raw data.npz")
    _require_file(label_npz, "raw node-label.npz")

    data = np.load(data_npz, mmap_mode="r")
    required = {"edge_index", "num_nodes_list", "num_edges_list"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"{data_npz} is missing required keys: {sorted(missing)}")

    if "node_feat" not in data.files:
        print("[warn] data.npz has no 'node_feat' key. Dataset may be incomplete.")

    labels = np.load(label_npz, mmap_mode="r")
    if "node_label" not in labels.files:
        raise RuntimeError(f"{label_npz} is missing required key: 'node_label'")


def _check_split(dataset_dir: Path) -> None:
    split_dir = dataset_dir / "split" / SPLIT_NAME
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    split_dict = split_dir / "split_dict.pt"
    if split_dict.exists():
        return

    for fname in ("train.csv.gz", "valid.csv.gz", "test.csv.gz"):
        _require_file(split_dir / fname, f"split file {fname}")


def _check_release(dataset_dir: Path) -> None:
    release = dataset_dir / "RELEASE_v1.txt"
    if not release.exists():
        print(f"[warn] Release marker not found: {release}")


def _process_dataset(root: Path) -> None:
    if PygNodePropPredDataset is None:
        raise RuntimeError("ogb is not installed. Please install ogb to proceed.")
    dataset = PygNodePropPredDataset(name=DATASET_NAME, root=str(root))
    processed_path = Path(dataset.processed_paths[0])
    print(f"[done] Processed dataset written to: {processed_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ogbn-papers100M PyG processed data file.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"Dataset root directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate raw and split files without processing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = _resolve_root(args.root)
    dataset_dir = _select_dataset_dir(root)

    print(f"[info] Using dataset root: {root}")
    print(f"[info] Checking dataset directory: {dataset_dir}")

    try:
        _check_raw(dataset_dir)
        _check_split(dataset_dir)
        _check_release(dataset_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.check_only:
        print("[done] Raw and split files look OK.")
        return

    _process_dataset(root)


if __name__ == "__main__":
    main()
