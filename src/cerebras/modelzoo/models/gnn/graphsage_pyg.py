import argparse
import os
import sys
import torch
from cerebras.modelzoo.models.gnn.pyg_gnn.utils import set_seed, load_cfg, setup_ddp, cleanup_ddp, ensure_pickle_friendly_load
from cerebras.modelzoo.models.gnn.pyg_gnn.cagnet_shim import destroy_cagnet_groups, clear_cagnet_caches
from cerebras.modelzoo.models.gnn.pyg_gnn.data import load_dataset, make_loaders, check_pyg_lib, _resolve_dataset_profile
from cerebras.modelzoo.models.gnn.pyg_gnn.model import get_model
from cerebras.modelzoo.models.gnn.pyg_gnn.train import train_model
from cerebras.modelzoo.models.gnn.pyg_gnn.caching import GraphCache
from torch_geometric.data import Data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    ap.add_argument("--cagnet-rows", type=int, default=1, help="CAGNET grid rows")
    ap.add_argument("--cagnet-cols", type=int, default=1, help="CAGNET grid cols")
    ap.add_argument("--cagnet-rep", type=int, default=1, help="CAGNET replication factor")
    ap.add_argument("--force-cagnet", action="store_true", help="Force usage of CagnetSAGE even if topology is 1x1x1")
    ap.add_argument("--use-partitions", action="store_true", help="Load offline partitions for training")
    ap.add_argument("--deterministic", action="store_true", help="Enable strict determinism (slower, but reproducible)")

    args = ap.parse_args()
    ensure_pickle_friendly_load()

    # Initialize DDP
    rank, world_size, device = setup_ddp()

    if rank == 0:
        if world_size > 1:
            print(f"[Info] Running in Multi-GPU DDP mode. World Size: {world_size}")
            print(f"[Info] Master Rank: {rank}, Device: {device}")
        else:
            print(f"[Info] Running in Single-Process mode (CPU or Single GPU).")
            print(f"[Info] Device: {device}")

    try:
        cfg = load_cfg(args.config)
        init = cfg["trainer"]["init"]
        model_dir = init["model_dir"]
        
        # Only rank 0 should probably create directories if they might conflict, 
        # but os.makedirs(exist_ok=True) is relatively safe.
        if rank == 0:
            os.makedirs(model_dir, exist_ok=True)

        seed = init["seed"]
        set_seed(seed, deterministic=args.deterministic)

        # ---- Dataset ----
        train_c = cfg["trainer"]["fit"]["train_dataloader"]
        dataset_profile = _resolve_dataset_profile(train_c)
        
        if args.use_partitions:
            from cerebras.modelzoo.models.gnn.pyg_gnn.data import load_dist_partition
            from cerebras.modelzoo.models.gnn.tools.partition_dataset import get_partition_path

            dataset_name = dataset_profile.get("dataset_name", train_c.get("dataset"))
            data_dir = dataset_profile.get("data_dir")
            if data_dir is None:
                print("[warn] data_dir is missing from dataset_profiles; cannot load partitions.", file=sys.stderr)
                raise SystemExit(1)
            part_dir = get_partition_path(dataset_name, data_dir, world_size)
            
            data, split_idx = load_dist_partition(str(part_dir), rank)
            print(f"[Info] Loaded partition {rank} from {part_dir}")
        
        else:
            data, split_idx = load_dataset(dataset_profile)

        if not args.use_partitions:
            check_pyg_lib()
        
        # Initialize GraphCache
        # Get caching percent from config, default to 0.0 to disable auto-caching
        train_dataloader_cfg = cfg.get("trainer", {}).get("fit", {}).get("train_dataloader", {})
        caching_percent = train_dataloader_cfg.get("caching_percent", 0.0)
        
        is_dist = isinstance(data, tuple)
        
        if not is_dist:
            cache = GraphCache(data, device, percent=caching_percent)
            
            # Create loader_data without x to avoid duplicate fetching
            loader_data = Data()
            for k, v in data:
                if k != "x":
                    loader_data[k] = v
            loader_data.num_nodes = data.num_nodes
            num_nodes = data.num_nodes
        else:
            # Distributed mode with partitions: direct loading
            # TODO: Add cache support for partitioned feature stores.
            cache = None
            loader_data = data # Pass tuple directly to make_loaders
            num_nodes = -1
        
        # make_loaders now supports rank/world_size
        loaders = make_loaders(loader_data, split_idx, cfg, rank=rank, world_size=world_size)

        # ---- Model ----
        model = get_model(cfg, args, num_nodes=num_nodes).to(device)

        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # DDP wrapper
            model = DDP(model, device_ids=[rank])
            print(f"[ddp] Rank {rank}: Wrapped model in DDP")

        if hasattr(torch, "compile") and not os.getenv("NO_COMPILE"):
            model = torch.compile(model)

        # ---- Train ----
        train_model(cfg, model, loaders, data, split_idx, device, rank=rank, world_size=world_size, cache=cache)
        
    finally:
        destroy_cagnet_groups()
        clear_cagnet_caches()
        cleanup_ddp()

if __name__ == "__main__":
    main()
