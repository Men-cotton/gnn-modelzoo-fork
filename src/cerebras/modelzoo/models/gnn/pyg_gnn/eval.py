import argparse
import os
import torch
import torch.distributed as dist
from cerebras.modelzoo.models.gnn.pyg_gnn.utils import set_seed, load_cfg
from cerebras.modelzoo.models.gnn.pyg_gnn.data import load_dataset, make_loaders, check_pyg_lib, _resolve_dataset_profile
from cerebras.modelzoo.models.gnn.pyg_gnn.model import get_model

@torch.no_grad()
def evaluate(model, loader, device, cache=None):
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if cache is not None:
             batch.x = cache.fetch(batch.n_id)
        # GraphSAGE with NeighborLoader: pass batch_size to get only seed nodes
        out = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
        out = out[: batch.batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[: batch.batch_size].view(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
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
    correct = (pred == y).sum().item()
    return correct / max(y.numel(), 1)

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
    
    check_pyg_lib()
    # We only need val loader for evaluation
    # But make_loaders returns both. Let's just use make_loaders for simplicity or manually create val loader.
    # make_loaders requires 'fit' section in config which might be present.
    # Let's reuse make_loaders to be consistent.
    _, val_loader = make_loaders(data, split_idx, cfg)

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
    acc = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
