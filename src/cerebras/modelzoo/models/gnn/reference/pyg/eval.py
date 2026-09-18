import argparse
import os
import torch
import torch.distributed as dist
from cerebras.modelzoo.models.gnn.reference.pyg.utils import set_seed, load_cfg
from cerebras.modelzoo.models.gnn.reference.pyg.data import (
    load_dataset,
    make_validation_loader,
    check_pyg_lib,
    _resolve_dataset_profile,
)
from cerebras.modelzoo.models.gnn.reference.pyg.model import get_model
from cerebras.modelzoo.models.gnn.gpu_policy import precision_dtype


@torch.no_grad()
def evaluate(model, loader, device, cache=None, dtype=torch.float32):
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            if hasattr(batch, "node_idx"):
                out = model(batch.x, batch.edge_index)
                node_idx = batch.node_idx
                out = out.index_select(0, node_idx)
                y = batch.y.index_select(0, node_idx).view(-1)
            else:
                if cache is not None:
                    batch.x = cache.fetch(batch.n_id)
                out = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
                out = out[: batch.batch_size]
                y = batch.y[: batch.batch_size].view(-1)
        pred = out.argmax(dim=-1)
        valid = y != -100
        correct += ((pred == y) & valid).sum().item()
        total += valid.sum().item()
    if dist.is_available() and dist.is_initialized():
        counts = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        correct = int(counts[0].item())
        total = int(counts[1].item())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_full_batch(model, data, node_idx, device):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[node_idx].argmax(dim=-1)
    y = data.y[node_idx].view(-1)
    valid = y != -100
    correct = ((pred == y) & valid).sum().item()
    return correct / max(valid.sum().item(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    ap.add_argument("--checkpoint", required=True, help="path to checkpoint")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # Seed
    seed = cfg["trainer"]["init"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_c = cfg["trainer"]["validate"]["val_dataloader"]
    dataset_profile = _resolve_dataset_profile(val_c)
    data, split_idx = load_dataset(dataset_profile)

    if val_c.get("sampling_mode", "neighbor") == "neighbor":
        check_pyg_lib()
    val_loader = make_validation_loader(data, split_idx, cfg)

    # Model
    model = get_model(cfg).to(device)

    # Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state"]
    # Fix for torch.compile adding _orig_mod prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # Evaluate
    init = cfg["trainer"]["init"]
    task = init["model"].get("task", init["model"])
    dtype = precision_dtype(
        init,
        default=(torch.float16 if task.get("to_float16", False) else torch.float32),
    )
    acc = evaluate(model, val_loader, device, dtype=dtype)
    print(f"Validation Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
