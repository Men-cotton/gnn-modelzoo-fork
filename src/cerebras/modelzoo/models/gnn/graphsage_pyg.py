import argparse
import os
import torch
from cerebras.modelzoo.models.gnn.pyg_gnn.utils import set_seed, load_cfg
from cerebras.modelzoo.models.gnn.pyg_gnn.data import load_dataset, make_loaders, check_pyg_lib, _resolve_dataset_profile
from cerebras.modelzoo.models.gnn.pyg_gnn.model import get_model
from cerebras.modelzoo.models.gnn.pyg_gnn.train import train_model

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
    data, split_idx = load_dataset(dataset_profile)

    check_pyg_lib()
    loaders = make_loaders(data, split_idx, cfg)

    # ---- Model ----
    model = get_model(cfg).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # ---- Train ----
    train_model(cfg, model, loaders, data, split_idx, device)


if __name__ == "__main__":
    main()
