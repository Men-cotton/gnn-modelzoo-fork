import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from .cagnet_shim import spmm_15d_or_2d, make_process_grid, spmm_supports_half_precision

def _row_norm(edge_index, N, device):
    src, dst = edge_index
    row = dst; col = src
    ones = torch.ones(row.numel(), device=device)
    deg = torch.bincount(row, minlength=int(N)).clamp_min_(1)
    val = ones / deg[row]
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (int(N), int(N)), device=device)
    return A.coalesce()

def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class CagnetSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_nodes: int,
        rows: int = 1,
        cols: int = 1,
        rep: int = 1,
        dropout: float = 0.5,
        force_cagnet: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)

        self.rows = rows
        self.cols = cols
        self.rep = rep
        self.dropout = dropout
        self.force_cagnet = force_cagnet

        self._grid_ready = False
        self.row_group: Optional[dist.ProcessGroup] = None
        self.col_group: Optional[dist.ProcessGroup] = None
        self._full_adj_cache = {}

        self.lin_neigh1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_self1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_neigh2 = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_self2 = torch.nn.Linear(hidden_channels, out_channels)

        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=dropout)

    def _maybe_init_grid(self):
        if self._grid_ready:
            return
        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        expected = self.rows * self.cols * max(1, self.rep)
        if expected != world:
            if self.rep != 1 and self.rows * self.cols == world:
                self.rep = 1
                if is_master():
                    print(
                        f"[CAGNET] rows*cols*rep={expected} != world_size={world}; "
                        "falling back to replication=1 topology."
                    )
            else:
                # Warning instead of Error to allow single-gpu testing if mismatch
                if is_master():
                    print(
                        f"[CAGNET] Warning: topology mismatch rows*cols*rep={expected}, world_size={world}. "
                        "This might crash if using distributed SpMM."
                    )
        
        # Only try to make grid if dist is initialized
        if dist.is_available() and dist.is_initialized():
            self.row_group, self.col_group = make_process_grid(
                world, self.rows, self.cols, self.rep
            )
        else:
            self.row_group, self.col_group = None, None
            
        self._grid_ready = True

    def _spmm(
        self, A: torch.Tensor, X: torch.Tensor, *, is_full_batch: bool
    ) -> torch.Tensor:
        self._maybe_init_grid()

        use_fp32 = (X.dtype in (torch.float16, torch.bfloat16)) and (
            not spmm_supports_half_precision()
        )
        if use_fp32:
            A32 = A.float()
            X32 = X.float()
        else:
            A32 = A
            X32 = X

        out32 = spmm_15d_or_2d(
            A32,
            X32,
            self.row_group,
            self.col_group,
            self.rep,
            is_full_batch=is_full_batch,
            force_cagnet=self.force_cagnet and (not is_full_batch),
        )

        if use_fp32:
            out = out32.to(dtype=X.dtype)
        else:
            out = out32

        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        N = x.size(0)
        dev = x.device
        edge_index = edge_index.to(dev, non_blocking=True)

        is_full_batch = (N == self.num_nodes)

        if is_full_batch:
            if dev not in self._full_adj_cache:
                self._full_adj_cache[dev] = _row_norm(edge_index, N, dev)
            A = self._full_adj_cache[dev]
        else:
            A = _row_norm(edge_index, N, dev)

        h1 = self.lin_neigh1(self._spmm(A, x, is_full_batch=is_full_batch)) + self.lin_self1(x)
        h1 = self.act(h1)
        h1 = self.drop(h1)

        h2 = self.lin_neigh2(self._spmm(A, h1, is_full_batch=is_full_batch)) + self.lin_self2(h1)

        return h2
