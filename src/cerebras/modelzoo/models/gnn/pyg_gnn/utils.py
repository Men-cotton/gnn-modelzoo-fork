import random
import os
import numpy as np
import torch
import torch.distributed as dist
import yaml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_ddp():
    """
    Initializes DDP if appropriate environment variables are set.
    Returns:
        rank (int): Global rank
        world_size (int): Total number of processes
        device (torch.device): Assigned device
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Initialize the process group
        dist.init_process_group(backend="nccl")
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        print(f"[ddp] Initialized: rank={rank}, world_size={world_size}, device={device}")
        return rank, world_size, device
    else:
        # Fallback for single GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ddp] Single-process mode: device={device}")
        return 0, 1, device

def cleanup_ddp():
    """Destroys the process group if it exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
