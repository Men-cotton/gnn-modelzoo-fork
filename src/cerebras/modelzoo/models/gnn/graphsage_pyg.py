# graphsage_pyg.py
import os, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GraphSAGE
from ogb.nodeproppred import PygNodePropPredDataset
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr, BaseData
import yaml
from torch_geometric.datasets import Reddit, Planetoid

_SAFE_GLOBALS = [Data, DataTensorAttr, DataEdgeAttr, BaseData, Data]
add_safe_globals(_SAFE_GLOBALS)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def _has_neighbor_sampling_support():
    try:
        import pyg_lib  # type: ignore  # noqa: F401

        return True
    except ImportError:
        try:
            import torch_sparse  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False


def _check_dataset_exists(data_dir, dataset_name):
    """Checks if the dataset is already processed and available locally."""
    import os.path as osp

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

def _load_dataset(profile):
    dataset_name = profile["dataset_name"]
    data_dir = profile["data_dir"]
    
    # Resolve data_dir relative to this script if it is relative
    if not os.path.isabs(data_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(base_dir, data_dir))
        print(f"[loader] Resolved data_dir to: {data_dir}")

    _check_dataset_exists(data_dir, dataset_name)

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


def make_loaders(data, split_idx, cfg):
    # 参照するブロック
    fit = cfg["trainer"]["fit"]
    train_c = fit["train_dataloader"]
    val_c = cfg["trainer"]["validate"]["val_dataloader"]

    train_profile = _resolve_dataset_profile(train_c)
    val_profile = _resolve_dataset_profile(val_c)

    train_loader = NeighborLoader(
        data,
        input_nodes=_resolve_split(split_idx, train_c.get("split", "train")),
        num_neighbors=_get_fanouts(train_c, train_profile),
        batch_size=train_c["batch_size"],
        shuffle=train_c["shuffle"],
        drop_last=train_c["drop_last_batch"],
        num_workers=train_c["num_workers"],
        generator=_build_generator(train_c),
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
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        # GraphSAGEはNeighborLoaderと併用時、batch_sizeを渡すと先頭seedノード分のみ返せる
        out = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
        out = out[: batch.batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[: batch.batch_size].view(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_full_batch(model, data, node_idx, device):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[node_idx].argmax(dim=-1)
    y = data.y[node_idx].view(-1)
    correct = (pred == y).sum().item()
    return correct / max(y.numel(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    init = cfg["trainer"]["init"]
    model_dir = init["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    seed = init["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    train_c = cfg["trainer"]["fit"]["train_dataloader"]
    dataset_profile = _resolve_dataset_profile(train_c)
    data, split_idx = _load_dataset(dataset_profile)

    use_neighbor_loader = _has_neighbor_sampling_support()
    if use_neighbor_loader:
        loaders = make_loaders(data, split_idx, cfg)
        train_loader, val_loader = loaders
    else:
        train_loader = None
        val_loader = None
        print(
            "[loader] Neither 'pyg-lib' nor 'torch-sparse' is available; "
            "falling back to full-batch training."
        )

    # ---- Model ----
    m = init["model"]
    model = GraphSAGE(
        in_channels=m["n_feat"],
        hidden_channels=m["graphsage_hidden_dim"],
        num_layers=m["graphsage_num_layers"],
        out_channels=m["n_class"],
        dropout=m["graphsage_dropout"],
        aggr=m["graphsage_aggregator"],  # SAGEConvのaggrへフォワード
    ).to(device)

    # ---- Optimizer ----
    optconf = init["optimizer"]["AdamW"]
    optimizer = AdamW(
        model.parameters(),
        lr=optconf["learning_rate"],
        weight_decay=optconf["weight_decay"],
    )

    # ---- AMP (fp16) ----
    use_amp = bool(m.get("to_float16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Config Options ----
    compute_eval_metrics = m.get("compute_eval_metrics", True)
    disable_log_softmax = m.get("disable_log_softmax", False)

    # ---- Loop ----
    loop = init["loop"]
    max_steps = int(loop["max_steps"])
    steps_per_epoch = int(loop["steps_per_epoch"])
    eval_frequency = int(loop["eval_frequency"])
    grad_accum = int(loop.get("grad_accum_steps", 1))

    log_steps = cfg["trainer"]["init"]["logging"]["log_steps"]

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    if use_neighbor_loader:
        train_iter = iter(train_loader)
    else:
        data = data.to(device)
        train_nodes = _resolve_split(split_idx, train_c.get("split", "train")).to(device)
        val_cfg = cfg["trainer"]["validate"]["val_dataloader"]
        val_nodes = _resolve_split(split_idx, val_cfg.get("split", "val")).to(device)

    while step < max_steps:
        if use_neighbor_loader:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
                logits = logits[: batch.batch_size]
                y = batch.y[: batch.batch_size]
                if not disable_log_softmax:
                    logits = F.log_softmax(logits, dim=-1)
                    loss = F.nll_loss(logits, y)
                else:
                    loss = F.cross_entropy(logits, y)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(data.x, data.edge_index)
                y = data.y[train_nodes]
                if not disable_log_softmax:
                    logits = F.log_softmax(logits[train_nodes], dim=-1)
                    loss = F.nll_loss(logits, y)
                else:
                    loss = F.cross_entropy(logits[train_nodes], y)

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % log_steps == 0:
            avg = running_loss / log_steps
            print(f"[step {step:04d}] loss={avg:.4f}")
            running_loss = 0.0

        if step % steps_per_epoch == 0:
            epoch += 1

        if compute_eval_metrics and (step % eval_frequency == 0 or step == max_steps):
            if use_neighbor_loader:
                val_acc = evaluate(model, val_loader, device)
            else:
                val_acc = evaluate_full_batch(model, data, val_nodes, device)
            print(f"[eval @ step {step}] val_acc={val_acc:.4f}")

    ckpt = os.path.join(model_dir, "last.pt")
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
