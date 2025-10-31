from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:  # pragma: no cover - Optional dependency
    PygNodePropPredDataset = None  # type: ignore[assignment]

from torch_geometric.datasets import Planetoid, Reddit

DEFAULT_ROOT = Path("./data/datasets")
REDDIT_URL = "https://data.dgl.ai/dataset/reddit.zip"
REDDIT_ZIP_NAME = "reddit.zip"
REDDIT_SHA256 = "9a16353c28f8ddd07148fc5ac9b57b818d7911ea0fbe9052d66d49fc32b372bf"
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MiB

_DATASET_ALIASES: Dict[str, str] = {
    "pubmed": "pubmed",
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn_arxiv": "ogbn-arxiv",
    "ogbn-mag": "ogbn-mag",
    "ogbn_mag": "ogbn-mag",
    "ogbn-products": "ogbn-products",
    "ogbn_products": "ogbn-products",
    "ogbn-papers100m": "ogbn-papers100m",
    "ogbn_papers100m": "ogbn-papers100m",
    "mag240m": "mag240m",
    "reddit": "reddit",
}

ALL_DATASETS = [
    "pubmed",
    "reddit",
    "ogbn-arxiv",
    "ogbn-mag",
    "ogbn-products",
    "ogbn-papers100m",
    "mag240m",
]


def _normalize_dataset_key(name: str) -> str:
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    return _DATASET_ALIASES.get(key, key)


def _request_download(dataset_name: str) -> bool:
    prompt = (
        f"[{dataset_name}] Dataset not found locally. "
        "Download now? Enter y to confirm (default: n): "
    )
    try:
        response = input(prompt)
    except EOFError:
        print(f"[{dataset_name}] No response received; skipping download.")
        return False

    if response.strip().lower() in ("y", "yes"):
        return True

    print(f"[{dataset_name}] Skipping download at user request.")
    return False


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(url: str, dest_path: Path, *, expected_sha256: Optional[str] = None) -> None:
    dest_path = dest_path.resolve()
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if expected_sha256:
            current_hash = _compute_sha256(dest_path)
            if current_hash == expected_sha256:
                print(f"[download] Found existing file with matching checksum: {dest_path}")
                temp_path.unlink(missing_ok=True)
                return
            print(f"[download] Existing file has wrong checksum ({current_hash}); re-downloading.")
        dest_path.unlink()

    download_complete = False
    while not download_complete:
        existing_size = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(f"[download] Resuming download from byte {existing_size}.")
        else:
            mode = "wb"
            print(f"[download] Starting download: {url}")

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response, temp_path.open(mode) as out_file:
                status = getattr(response, "status", None) or response.getcode()
                if existing_size > 0 and status == 200:
                    # Server did not honor range request; restart download.
                    print("[download] Range header ignored. Restarting download from scratch.")
                    temp_path.unlink(missing_ok=True)
                    continue

                headers = response.headers
                total_size = None
                if headers:
                    content_range = headers.get("Content-Range")
                    if content_range and "/" in content_range:
                        total_part = content_range.rsplit("/", 1)[-1]
                        if total_part.isdigit():
                            total_size = int(total_part)
                    elif headers.get("Content-Length"):
                        length = int(headers["Content-Length"])
                        total_size = length + (existing_size if status == 206 else 0)

                downloaded_bytes = existing_size
                last_percent = -1

                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)

                    downloaded_bytes += len(chunk)
                    if total_size:
                        percent = int(downloaded_bytes * 100 / total_size)
                        if percent != last_percent:
                            print(
                                f"[download] {percent}% ({downloaded_bytes:,}/{total_size:,} bytes)",
                                end="\r",
                                flush=True,
                            )
                            last_percent = percent
                    else:
                        print(
                            f"[download] Downloaded {downloaded_bytes:,} bytes",
                            end="\r",
                            flush=True,
                        )
                if total_size:
                    # ensure we print 100% on completion
                    print(
                        f"[download] 100% ({downloaded_bytes:,}/{total_size:,} bytes)",
                        flush=True,
                    )
                else:
                    print(flush=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and temp_path.exists():
                # Requested range not satisfiable -> file already complete.
                print("[download] Remote reported complete file; finishing up.")
            else:
                raise

        download_complete = True

    os.replace(temp_path, dest_path)
    if expected_sha256:
        actual_sha256 = _compute_sha256(dest_path)
        if actual_sha256 != expected_sha256:
            dest_path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {dest_path}. Expected {expected_sha256}, got {actual_sha256}."
            )
        print(f"[download] Checksum verified for {dest_path}.")


def download_pubmed(root_dir: Path) -> None:
    dataset_name = "PubMed"
    dataset_root = root_dir / dataset_name
    processed_path = dataset_root / "processed" / "data.pt"
    if processed_path.exists():
        print(f"[pubmed] Dataset already prepared at {processed_path}.")
        return
    if not _request_download(dataset_name):
        return

    print(f"[pubmed] Downloading and processing '{dataset_name}' into '{root_dir}'.")
    dataset = Planetoid(root=str(root_dir), name=dataset_name, split="public")
    print(f"[pubmed] Raw files located at: {dataset.raw_dir}")
    print(f"[pubmed] Processed data stored at: {dataset.processed_dir}")
    if processed_path.exists():
        print(f"[pubmed] Confirmed processed file: {processed_path}")
    else:
        print("[pubmed] Warning: processed file data.pt not found.")


def download_reddit(root_dir: Path) -> None:
    dataset_name = "Reddit"
    dataset_dir = root_dir / dataset_name
    processed_path = dataset_dir / "processed" / "data.pt"
    if processed_path.exists():
        print(f"[reddit] Dataset already prepared at {processed_path}.")
        return
    if not _request_download(dataset_name):
        return

    raw_dir = dataset_dir / "raw"
    archive_path = raw_dir / REDDIT_ZIP_NAME
    print(f"[reddit] Preparing download directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        _download_with_resume(REDDIT_URL, archive_path, expected_sha256=REDDIT_SHA256)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[reddit] Failed to download archive: {exc}")
        raise

    print(f"[reddit] Extracting archive to {raw_dir}")
    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        zip_handle.extractall(raw_dir)

    print("[reddit] Generating processed dataset via PyG.")
    dataset = Reddit(root=str(dataset_dir))
    print(f"[reddit] Raw files located at: {dataset.raw_dir}")
    print(f"[reddit] Processed data stored at: {dataset.processed_dir}")
    if processed_path.exists():
        print(f"[reddit] Confirmed processed file: {processed_path}")
    else:
        print("[reddit] Warning: processed file data.pt not found.")


def download_ogb_dataset(dataset_name: str, root_dir: Path) -> None:
    if PygNodePropPredDataset is None:
        raise ImportError(
            "ogb is not installed. Install it with `uv pip install ogb` or "
            "`pip install ogb` before downloading OGB datasets."
        )
    dataset_dir = root_dir / dataset_name
    dataset_subdir = dataset_dir / dataset_name.replace("-", "_")
    processed_flag = dataset_subdir / "processed"
    if processed_flag.exists():
        print(f"[{dataset_name}] Dataset already prepared under {processed_flag}.")
        return
    if not _request_download(dataset_name):
        return

    print(f"[{dataset_name}] Downloading with PygNodePropPredDataset into {dataset_dir}.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = PygNodePropPredDataset(name=dataset_name, root=str(dataset_dir))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Failed to download OGB dataset '{dataset_name}'."
        ) from exc

    split_idx = dataset.get_idx_split()
    print(
        f"[{dataset_name}] Raw directory: {dataset.raw_dir}\n"
        f"[{dataset_name}] Processed directory: {dataset.processed_dir}\n"
        f"[{dataset_name}] Split sizes -> "
        f"train: {len(split_idx.get('train', []))}, "
        f"val: {len(split_idx.get('valid', []))}, "
        f"test: {len(split_idx.get('test', []))}"
    )


def download_mag240m_placeholder(root_dir: Path) -> None:
    dataset_name = "MAG240M"
    dataset_dir = root_dir / dataset_name
    processed_flag = dataset_dir / dataset_name / "processed"
    if processed_flag.exists():
        print(f"[mag240m] Dataset already prepared under {processed_flag}.")
        return
    if _request_download(dataset_name):
        print(
            "[mag240m] Automated download is not supported. "
            "Please follow the official OGB instructions to stage this dataset manually."
        )


def _build_handler_registry() -> Dict[str, Callable[[Path], None]]:
    return {
        "pubmed": download_pubmed,
        "reddit": download_reddit,
        "ogbn-arxiv": partial(download_ogb_dataset, "ogbn-arxiv"),
        "ogbn-mag": partial(download_ogb_dataset, "ogbn-mag"),
        "ogbn-products": partial(download_ogb_dataset, "ogbn-products"),
        "ogbn-papers100m": partial(download_ogb_dataset, "ogbn-papers100m"),
        "mag240m": download_mag240m_placeholder,
    }


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GNN datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Destination directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Datasets to fetch. Choices include "
            "'pubmed', 'reddit', 'ogbn-arxiv', 'ogbn-mag', "
            "'ogbn-products', 'ogbn-papers100m', 'mag240m'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    root = args.root
    handlers = _build_handler_registry()

    print(f"[main] Using dataset root: {root.resolve()}")
    failures = []

    dataset_names = args.datasets or ALL_DATASETS

    for raw_name in dataset_names:
        dataset_key = _normalize_dataset_key(raw_name)
        handler = handlers.get(dataset_key)
        if handler is None:
            print(f"[main] Skipping unknown dataset '{raw_name}'.")
            continue
        print(f"[main] Processing dataset '{dataset_key}'.")
        try:
            handler(root)
        except NotImplementedError as exc:
            print(f"[main] Skipped '{dataset_key}': {exc}")
        except Exception as exc:
            failures.append(dataset_key)
            print(
                f"[main] Failed to prepare dataset '{dataset_key}': {exc}",
                file=sys.stderr,
            )

    if failures:
        print(f"[main] Failed datasets: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)

    print("[main] All requested datasets are ready or skipped intentionally.")


if __name__ == "__main__":
    main()
