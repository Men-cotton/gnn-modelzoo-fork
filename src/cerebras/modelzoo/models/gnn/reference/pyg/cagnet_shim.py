# SPDX-License-Identifier: Apache-2.0
# Portions of this file are derived from the CAGNET project
# (Copyright (c) 2020, The Regents of the University of California,
# Berkeley) released under a BSD-3-Clause style license.
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.distributed as dist
import os

_HAS_SPMM_GPU = False
_spmm_gpu = None
# Prioritize using integrated sparse_coo_tensor_cpp build
for _mod_name in ("sparse_coo_tensor_cpp",):
    try:
        _mod = __import__(_mod_name)
        _spmm_gpu = _mod.spmm_gpu  # type: ignore[attr-defined]
        _HAS_SPMM_GPU = True
        break
    except Exception:
        continue

_WARNED = set()
_SPLIT_CACHE: Dict[int, dict] = {}
_FEATURE_SPLIT_CACHE: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
# Cache broadcast results for A/X blocks (assuming fixed full-batch size, managed per X)
_AX_BCAST_CACHE: Dict[int, dict] = {}
_CAGNET_STRICT = bool(int(os.getenv("CAGNET_STRICT", "0")))


def clear_cagnet_caches() -> None:
    """Explicitly clear cached splits and broadcasted blocks (vendor-style reuse reset)."""
    _SPLIT_CACHE.clear()
    _FEATURE_SPLIT_CACHE.clear()
    _AX_BCAST_CACHE.clear()


def destroy_cagnet_groups() -> None:
    """Best-effort destroy of CAGNET-specific process groups to suppress leak warnings."""
    global _current_topology
    topo = _current_topology
    if topo is None:
        return
    for pg in (topo.row_group, topo.col_group, topo.rep_group, topo.plane_group):
        try:
            if pg is not None:
                dist.destroy_process_group(pg)
        except Exception:
            pass
    _current_topology = None


@dataclass
class _Topology:
    rows: int
    cols: int
    replication: int
    world_size: int
    row_group: Optional[dist.ProcessGroup] = None
    col_group: Optional[dist.ProcessGroup] = None
    rep_group: Optional[dist.ProcessGroup] = None
    plane_group: Optional[dist.ProcessGroup] = None

    @property
    def kind(self) -> str:
        """
        Rough topology classification.
        - rows>1, cols>1, replication>1 -> "3d"
        - rows>1, cols>1, replication==1 -> "2d"
        - rows==1, cols>1 -> "1d/1.5d"
        - Otherwise -> "none" (essentially data-parallel)
        """
        if self.world_size <= 1:
            return "none"
        if self.rows > 1 and self.cols > 1 and self.replication > 1:
            return "3d"
        if self.rows > 1 and self.cols > 1:
            return "2d"
        if self.cols > 1:
            return "1d/1.5d"
        return "none"


_current_topology: Optional[_Topology] = None


def _warn_once(msg: str) -> None:
    """Emit a warning only once per unique message."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    warnings.warn(f"[cagnet-shim] {msg}")


def spmm_supports_half_precision() -> bool:
    """
    Return True when the SpMM backend natively supports half/bf16.
    Default CUDA kernel for CAGNET assumes float32, so False.
    """
    return False


def _local_spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Use CAGNET's CUDA kernel if available, otherwise torch.sparse.mm."""
    if not X.is_contiguous():
        X = X.contiguous()
    if (
        _HAS_SPMM_GPU
        and A.is_cuda
        and X.is_cuda
        and A.dtype == torch.float32
        and X.dtype == torch.float32
    ):
        out = torch.zeros(A.size(0), X.size(1), device=X.device, dtype=X.dtype)
        _spmm_gpu(
            A.indices()[0].int(),
            A.indices()[1].int(),
            A.values().float(),
            int(A.size(0)),
            int(A.size(1)),
            X.contiguous(),
            out,
        )
        return out.contiguous()
    out = torch.sparse.mm(A, X)
    return out.contiguous() if not out.is_contiguous() else out


def make_process_grid(
    world_size: int,
    rows: int,
    cols: int,
    replication: int = 1,
):
    """
    Construct a rows x cols x replication CAGNET topology and
    return the row_group / col_group that this rank belongs to.
    """
    global _current_topology

    if replication < 1:
        replication = 1

    if (not dist.is_available()) or (not dist.is_initialized()):
        _current_topology = _Topology(rows, cols, replication, world_size)
        return None, None

    expected = rows * cols * replication
    if world_size != expected:
        if replication != 1 and rows * cols == world_size:
            _warn_once(
                "make_process_grid: rows*cols*replication != world_size; "
                "falling back to replication=1 (2D topology)."
            )
            replication = 1
            expected = rows * cols
        if world_size != expected:
            raise ValueError(
                f"make_process_grid: rows*cols*replication={expected} "
                f"does not match world_size={world_size}"
            )

    # For 3D, enforce near-cubic grid (similar to vendor checks)
    if replication > 1:
        cube = int(round(world_size ** (1.0 / 3.0)))
        if cube * cube * cube != world_size:
            _warn_once(
                "3D topology recommended on near-cubic world sizes; proceeding anyway."
            )

    if world_size <= 1 or (rows == 1 and cols == 1 and replication == 1):
        _current_topology = _Topology(rows, cols, replication, world_size)
        return None, None

    rank = dist.get_rank()
    plane_size = rows * cols
    rep_id = rank // plane_size
    in_plane = rank % plane_size
    row_id = in_plane // cols
    col_id = in_plane % cols

    my_row_pg = None
    my_col_pg = None
    my_rep_pg = None
    my_plane_pg = None

    # (A) plane groups: 1 per replication
    for rep in range(replication):
        members = list(range(rep * plane_size, (rep + 1) * plane_size))
        pg = dist.new_group(members)
        if rep == rep_id:
            my_plane_pg = pg

    # (B) row groups: replication * rows count
    for rep in range(replication):
        base = rep * plane_size
        for r in range(rows):
            members = [base + r * cols + c for c in range(cols)]
            pg = dist.new_group(members)
            if rep == rep_id and r == row_id:
                my_row_pg = pg

    # (C) col groups: replication * cols count
    for rep in range(replication):
        base = rep * plane_size
        for c in range(cols):
            members = [base + r * cols + c for r in range(rows)]
            pg = dist.new_group(members)
            if rep == rep_id and c == col_id:
                my_col_pg = pg

    # (D) rep groups: 1 per position in plane
    if replication > 1:
        for p in range(plane_size):
            members = [rep * plane_size + p for rep in range(replication)]
            pg = dist.new_group(members)
            if p == in_plane:
                my_rep_pg = pg

    _current_topology = _Topology(
        rows,
        cols,
        replication,
        world_size,
        row_group=my_row_pg,
        col_group=my_col_pg,
        rep_group=my_rep_pg,
        plane_group=my_plane_pg,
    )
    return my_row_pg, my_col_pg


def _split_coo_parts(
    A: torch.Tensor, parts: int, dim: int
) -> Tuple[List[torch.Tensor], List[int]]:
    """Partition a COO tensor along the specified dimension into `parts` chunks."""
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    N = A.size(dim)
    if parts <= 0:
        parts = 1
    step = int(math.ceil(N / parts))
    bounds = [0]
    for i in range(parts):
        bounds.append(min(N, bounds[-1] + step))
    parts_out: List[torch.Tensor] = []
    for i in range(parts):
        start = bounds[i]
        end = bounds[i + 1]
        if end <= start:
            shape = list(A.size())
            shape[dim] = 0
            empty_idx = torch.empty((2, 0), device=idx.device, dtype=idx.dtype)
            empty_val = torch.empty((0,), device=val.device, dtype=val.dtype)
            parts_out.append(
                torch.sparse_coo_tensor(
                    empty_idx, empty_val, size=tuple(shape), device=A.device
                ).coalesce()
            )
            continue
        mask = (idx[dim] >= start) & (idx[dim] < end)
        if mask.sum().item() == 0:
            shape = list(A.size())
            shape[dim] = end - start
            empty_idx = torch.empty((2, 0), device=idx.device, dtype=idx.dtype)
            empty_val = torch.empty((0,), device=val.device, dtype=val.dtype)
            parts_out.append(
                torch.sparse_coo_tensor(
                    empty_idx, empty_val, size=tuple(shape), device=A.device
                ).coalesce()
            )
            continue
        sub_idx = idx[:, mask].clone()
        sub_idx[dim] -= start
        shape = list(A.size())
        shape[dim] = end - start
        parts_out.append(
            torch.sparse_coo_tensor(
                sub_idx, val[mask].clone(), size=tuple(shape), device=A.device
            ).coalesce()
        )
    return parts_out, bounds


def _split_coo_cached_single(
    A: torch.Tensor, parts: int, dim: int, idx: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Cached variant of _split_coo_single to avoid repeated coalesce/slicing
    on fixed full-batch adjacency. Cache key: (tensor, device, parts, dim).
    """
    A = A.coalesce()
    cache = _SPLIT_CACHE.setdefault(int(A._cdata), {})
    key = (A.device, int(parts), int(dim))
    if key not in cache:
        cache[key] = _split_coo_parts(A, parts, dim)
    parts_out, bounds = cache[key]
    idx = max(0, min(idx, max(0, parts - 1)))
    start = bounds[idx]
    end = bounds[idx + 1]
    return parts_out[idx], (start, end)


def _split_coo_single(
    A: torch.Tensor, parts: int, dim: int, idx: int, *, use_cache: bool = True
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition a COO tensor and return only the idx-th chunk (with bounds)."""
    if use_cache:
        return _split_coo_cached_single(A, parts, dim, idx)
    # No cache: split in-place
    A = A.coalesce()
    parts_out, bounds = _split_coo_parts(A, parts, dim)
    idx = max(0, min(idx, max(0, parts - 1)))
    start = bounds[idx]
    end = bounds[idx + 1]
    return parts_out[idx], (start, end)


def _feature_chunk_bounds(total: int, parts: int) -> List[Tuple[int, int]]:
    """Return cached start/end bounds for splitting feature dim into `parts`."""
    if parts <= 0:
        parts = 1
    key = (int(total), int(parts))
    if key in _FEATURE_SPLIT_CACHE:
        return _FEATURE_SPLIT_CACHE[key]
    step = int(math.ceil(total / parts))
    bounds = []
    for i in range(parts):
        s = i * step
        e = min(total, s + step)
        bounds.append((s, e))
    _FEATURE_SPLIT_CACHE[key] = bounds
    return bounds


def spmm_15d_or_2d(
    A: torch.Tensor,
    X: torch.Tensor,
    row_group: Optional[dist.ProcessGroup] = None,
    col_group: Optional[dist.ProcessGroup] = None,
    replication: int = 1,
    *,
    is_full_batch: bool = False,
    rep_group: Optional[dist.ProcessGroup] = None,
    plane_group: Optional[dist.ProcessGroup] = None,
    weight: Optional[torch.Tensor] = None,
    force_cagnet: bool = False,
) -> torch.Tensor:
    """
    1D / 1.5D / 2D / 3D SpMM wrapper for CAGNET.

    - Perform communication-avoiding SpMM only when is_full_batch=True and row_group is valid.
    - Always use local SpMM when is_full_batch=False (e.g. NeighborLoader mini-batch).
    - Accepts 3D topology, but current kernel logic is same as 2D (TODO).
    """
    topo = _current_topology
    g_row = row_group or (topo.row_group if topo else None)
    g_col = col_group or (topo.col_group if topo else None)
    use_cache = bool(is_full_batch)

    # --- Stage 0: Uninitialized DDP / No PG / Not full-batch ---
    if (
        (not dist.is_available())
        or (not dist.is_initialized())
        or (g_row is None)
        or ((not is_full_batch) and (not force_cagnet))
    ):
        out_local = _local_spmm(A, X)
        if weight is not None:
            out_local = out_local.matmul(weight.t())
        return out_local

    # Check world_size of row_group
    try:
        pg_world = dist.get_world_size(g_row)
    except Exception:
        _warn_once("failed to query row_group world_size; falling back to local SpMM.")
        out_local = _local_spmm(A, X)
        if weight is not None:
            out_local = out_local.matmul(weight.t())
        return out_local

    if pg_world <= 1:
        # No point communicating if single process
        out_local = _local_spmm(A, X)
        if weight is not None:
            out_local = out_local.matmul(weight.t())
        return out_local

    topo = _current_topology
    topo_kind = topo.kind if topo is not None else "unknown"

    try:
        if topo_kind == "3d" and topo is not None and topo.replication > 1:
            return _spmm_process_grid_3d(A, X, topo, weight=weight, use_cache=use_cache)
        if topo_kind == "2d" and topo is not None and topo.rows > 1 and topo.cols > 1:
            return _spmm_process_grid_2d_ax(A, X, topo, use_cache=use_cache)
        return _spmm_process_grid(A, X, g_row, g_col, use_cache=use_cache)
    except Exception as e:
        _warn_once(f"grid SpMM failed ({e}); " "falling back to local SpMM only.")
        out = _local_spmm(A, X)
        if weight is not None:
            out = out.matmul(weight.t())
        if _CAGNET_STRICT:
            raise
        return out


def _spmm_process_grid(
    A: torch.Tensor,
    X: torch.Tensor,
    row_group: dist.ProcessGroup,
    col_group: Optional[dist.ProcessGroup],
    *,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    1.5D/2D SpMM based on CAGNET column splitting + row-wise AllReduce.

    - Split columns by number of row_group members (=cols) (each rank holds a column block per col_id).
    - Split rows by rows(=col_group world_size) and recover via all_gather in col_group if needed.
    - Aggregate column split contributions within row group via AllReduce.
    """
    cols = dist.get_world_size(row_group)
    col_id = dist.get_rank(row_group)
    rows = dist.get_world_size(col_group) if col_group is not None else 1
    row_id = dist.get_rank(col_group) if col_group is not None else 0

    # Row slice
    if rows > 1:
        row_part, (r_start, r_end) = _split_coo_single(
            A, rows, dim=0, idx=row_id, use_cache=use_cache
        )
    else:
        row_part, (r_start, r_end) = A, (0, A.size(0))

    # Build column slice
    part, (c_start, c_end) = _split_coo_single(
        row_part, cols, dim=1, idx=col_id, use_cache=use_cache
    )
    if part.device != X.device:
        part = part.to(device=X.device)

    x_slice = X[c_start:c_end].contiguous()
    # If dtype is half precision, promote to float32 for kernel call, then convert back
    target_dtype = x_slice.dtype
    use_fp32 = target_dtype in (torch.float16, torch.bfloat16, torch.float64)
    A_eff = part.float() if use_fp32 else part
    X_eff = x_slice.float() if use_fp32 else x_slice

    with torch.autocast(device_type="cuda", enabled=False):
        partial = _local_spmm(A_eff, X_eff)
    partial = partial.contiguous()
    if partial.dtype != target_dtype:
        partial = partial.to(dtype=target_dtype)

    # Aggregate column-wise split results within row group
    try:
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=row_group)
    except Exception as e:
        _warn_once(
            f"all_reduce failed in CAGNET SpMM ({e}); falling back to local result."
        )
        return partial

    # If row-split, restore by all_gather in col_group
    if rows > 1 and col_group is not None:
        rows_local = torch.tensor(
            [r_end - r_start], device=partial.device, dtype=torch.int64
        )
        rows_bufs = [torch.zeros_like(rows_local) for _ in range(rows)]
        try:
            dist.all_gather(rows_bufs, rows_local, group=col_group)
        except Exception as e:
            _warn_once(
                f"all_gather(row sizes) failed in CAGNET SpMM ({e}); returning row-sliced output."
            )
            return partial
        sizes = [int(t.item()) for t in rows_bufs]
        max_rows = max(sizes) if sizes else 0
        if max_rows == 0:
            return partial
        if partial.size(0) < max_rows:
            pad = max_rows - partial.size(0)
            partial_padded = torch.cat(
                [
                    partial,
                    torch.zeros(
                        pad, partial.size(1), device=partial.device, dtype=partial.dtype
                    ),
                ],
                dim=0,
            )
        else:
            partial_padded = partial
        gather_bufs = [
            torch.empty(
                max_rows, partial.size(1), device=partial.device, dtype=partial.dtype
            )
            for _ in range(rows)
        ]
        try:
            dist.all_gather(gather_bufs, partial_padded, group=col_group)
        except Exception as e:
            _warn_once(
                f"all_gather(row blocks) failed in CAGNET SpMM ({e}); returning row-sliced output."
            )
            return partial
        full_rows = torch.cat([buf[:sz] for buf, sz in zip(gather_bufs, sizes)], dim=0)
        return full_rows

    return partial


def _spmm_process_grid_2d_ax(
    A: torch.Tensor,
    X: torch.Tensor,
    topo: _Topology,
    *,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    2D SUMMA-style SpMM (AX) distributed.
    - Split A into blocks of rows x cols.
    - Cut X only by column blocks (assume all ranks hold X, broadcast in col_group if needed).
    - SUM column contributions in row_group.
    - all_gather row blocks in col_group (variable length) to restore (N, F) on each rank.
    """
    rows = max(1, topo.rows)
    cols = max(1, topo.cols)
    row_group = topo.row_group
    col_group = topo.col_group
    if row_group is None:
        return _local_spmm(A, X)

    # rank → (row_id, col_id)
    col_id = dist.get_rank(row_group)
    row_id = dist.get_rank(col_group) if (col_group is not None) else 0

    # Row blocks
    if rows > 1:
        row_part, (r_start, r_end) = _split_coo_single(
            A, rows, dim=0, idx=row_id, use_cache=use_cache
        )
    else:
        row_part, (r_start, r_end) = A, (0, A.size(0))

    # Column blocks
    part, (c_start, c_end) = _split_coo_single(
        row_part, cols, dim=1, idx=col_id, use_cache=use_cache
    )
    if part.device != X.device:
        part = part.to(device=X.device)

    # Input slice (assume all ranks hold X, owner broadcast not performed)
    x_slice = X[c_start:c_end].contiguous()
    target_dtype = x_slice.dtype
    use_fp32 = target_dtype in (torch.float16, torch.bfloat16, torch.float64)
    A_eff = part.float() if use_fp32 else part
    X_eff = x_slice.float() if use_fp32 else x_slice

    with torch.autocast(device_type="cuda", enabled=False):
        partial = _local_spmm(A_eff, X_eff)
    partial = partial.contiguous()
    if partial.dtype != target_dtype:
        partial = partial.to(dtype=target_dtype)

    # SUM column contributions in row_group
    try:
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=row_group)
    except Exception as e:
        _warn_once(
            f"all_reduce failed in 2D SpMM row_group ({e}); falling back to local row slice."
        )
        return partial

    # If rows>1, all_gather row blocks in col_group (variable length)
    if rows > 1 and col_group is not None:
        rows_local = torch.tensor(
            [r_end - r_start], device=partial.device, dtype=torch.int64
        )
        rows_bufs = [torch.zeros_like(rows_local) for _ in range(rows)]
        try:
            dist.all_gather(rows_bufs, rows_local, group=col_group)
        except Exception as e:
            _warn_once(
                f"all_gather(row sizes) failed in 2D SpMM ({e}); returning row slice."
            )
            return partial
        sizes = [int(t.item()) for t in rows_bufs]
        max_rows = max(sizes) if sizes else 0
        if max_rows == 0:
            return partial
        if partial.size(0) < max_rows:
            pad = max_rows - partial.size(0)
            partial_padded = torch.cat(
                [
                    partial,
                    torch.zeros(
                        pad, partial.size(1), device=partial.device, dtype=partial.dtype
                    ),
                ],
                dim=0,
            )
        else:
            partial_padded = partial

        gather_bufs = [
            torch.empty(
                max_rows, partial.size(1), device=partial.device, dtype=partial.dtype
            )
            for _ in range(rows)
        ]
        try:
            dist.all_gather(gather_bufs, partial_padded, group=col_group)
        except Exception as e:
            _warn_once(
                f"all_gather(row blocks) failed in 2D SpMM ({e}); returning row slice."
            )
            return partial
        full_rows = torch.cat([buf[:sz] for buf, sz in zip(gather_bufs, sizes)], dim=0)
        return full_rows

    return partial


def _spmm_process_grid_3d(
    A: torch.Tensor,
    X: torch.Tensor,
    topo: _Topology,
    weight: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Simplified SpMM for 3D topology.
    - Split rows by `rows`, cols by `cols`.
    - Aggregate partial SpMM for each col slice via SUM in row_group.
    - Split feature dimension by rep_group, restore via all_gather.
    Approximate original split3dspmm_sparse logic with concise Python control flow.
    """
    if topo.replication <= 1 or topo.row_group is None or topo.rep_group is None:
        return _spmm_process_grid(A, X, topo.row_group, topo.col_group)

    rows, cols = topo.rows, topo.cols
    rank = dist.get_rank()
    plane_size = rows * cols
    rc = rank % plane_size
    row_id = rc // cols
    col_id = rc % cols

    # Slice sparse matrix: row split -> col split (cached)
    row_part, (r_start, r_end) = _split_coo_single(
        A, rows, dim=0, idx=row_id, use_cache=use_cache
    )
    col_part, (c_start, c_end) = _split_coo_single(
        row_part, cols, dim=1, idx=col_id, use_cache=use_cache
    )
    if col_part.device != X.device:
        col_part = col_part.to(device=X.device)

    # Split feature dimension by rep (handled by rep_id)
    F = X.size(1)
    rep_world = dist.get_world_size(topo.rep_group)
    rep_rank = dist.get_rank(topo.rep_group)
    feat_bounds = _feature_chunk_bounds(F, rep_world)
    f_start, f_end = feat_bounds[rep_rank] if rep_rank < len(feat_bounds) else (0, 0)
    if f_end <= f_start:
        # Empty tensor if no features for this rank
        x_slice = X.new_zeros((c_end - c_start, 0))
    else:
        if use_cache:
            # Cache A/X slices (per X) following vendor broadcast reuse pattern
            x_cache = _AX_BCAST_CACHE.setdefault(int(X._cdata), {})
            key = (c_start, c_end, f_start, f_end, X.device, X.dtype)
            x_slice = x_cache.get(key)
            if x_slice is None:
                x_slice = X[c_start:c_end, f_start:f_end].contiguous()
                x_cache[key] = x_slice
        else:
            x_slice = X[c_start:c_end, f_start:f_end].contiguous()

    target_dtype = x_slice.dtype
    use_fp32 = target_dtype in (torch.float16, torch.bfloat16, torch.float64)
    A_eff = col_part.float() if use_fp32 else col_part
    X_eff = x_slice.float() if use_fp32 else x_slice

    with torch.autocast(device_type="cuda", enabled=False):
        partial = _local_spmm(A_eff, X_eff)
    partial = partial.contiguous()
    if partial.dtype != target_dtype:
        partial = partial.to(dtype=target_dtype)

    # Aggregate column contributions in row_group (SUM)
    if topo.row_group is not None:
        try:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=topo.row_group)
        except Exception as e:
            _warn_once(
                f"all_reduce failed in 3D SpMM row_group ({e}); falling back to local result."
            )
            return partial
    # all_gather row blocks in col_group (variable length) to restore (N, f_chunk)
    rows_local = torch.tensor(
        [r_end - r_start], device=partial.device, dtype=torch.int64
    )
    rows_bufs = [torch.zeros_like(rows_local) for _ in range(rows)]
    try:
        dist.all_gather(rows_bufs, rows_local, group=topo.col_group)
    except Exception as e:
        _warn_once(
            f"all_gather(row sizes) failed in 3D SpMM ({e}); returning row slice."
        )
        return partial
    sizes_row = [int(t.item()) for t in rows_bufs]
    max_rows = max(sizes_row) if sizes_row else 0
    if max_rows == 0:
        return partial
    if partial.size(0) < max_rows:
        pad = max_rows - partial.size(0)
        partial_padded = torch.cat(
            [
                partial,
                torch.zeros(
                    pad, partial.size(1), device=partial.device, dtype=partial.dtype
                ),
            ],
            dim=0,
        )
    else:
        partial_padded = partial
    gather_row_bufs = [
        torch.empty(
            max_rows, partial.size(1), device=partial.device, dtype=partial.dtype
        )
        for _ in range(rows)
    ]
    try:
        dist.all_gather(gather_row_bufs, partial_padded, group=topo.col_group)
    except Exception as e:
        _warn_once(
            f"all_gather(row blocks) failed in 3D SpMM ({e}); returning row slice."
        )
        return partial
    full_rows = torch.cat(
        [buf[:sz] for buf, sz in zip(gather_row_bufs, sizes_row)], dim=0
    )

    # Gather each chunk length in rep dimension (share size without padding)
    len_local = torch.tensor(
        [full_rows.size(1)], device=full_rows.device, dtype=torch.int64
    )
    len_bufs = [torch.zeros_like(len_local) for _ in range(rep_world)]
    try:
        dist.all_gather(len_bufs, len_local, group=topo.rep_group)
    except Exception as e:
        _warn_once(
            f"all_gather(len) failed in 3D SpMM ({e}); returning row-assembled output."
        )
        if weight is not None:
            return full_rows.matmul(weight.t())
        return full_rows
    sizes = [int(t.item()) for t in len_bufs]
    max_len = max(sizes) if sizes else 0
    if max_len == 0:
        if weight is not None:
            return full_rows.matmul(weight.t())
        return full_rows

    # all_gather in rep dimension (variable length)
    if full_rows.size(1) < max_len:
        pad_cols = max_len - full_rows.size(1)
        full_rows_padded = torch.cat(
            [
                full_rows,
                torch.zeros(
                    full_rows.size(0),
                    pad_cols,
                    device=full_rows.device,
                    dtype=full_rows.dtype,
                ),
            ],
            dim=1,
        )
    else:
        full_rows_padded = full_rows
    gather_feat_bufs = [
        torch.empty(
            full_rows.size(0), max_len, device=full_rows.device, dtype=full_rows.dtype
        )
        for _ in range(rep_world)
    ]
    try:
        dist.all_gather(gather_feat_bufs, full_rows_padded, group=topo.rep_group)
    except Exception as e:
        _warn_once(
            f"all_gather(data) failed in 3D SpMM ({e}); returning row-assembled output."
        )
        if weight is not None:
            return full_rows.matmul(weight.t())
        return full_rows

    full_feat = torch.cat(
        [buf[:, :sz] for buf, sz in zip(gather_feat_bufs, sizes)], dim=1
    )
    if weight is not None:
        return full_feat.matmul(weight.t())
    return full_feat
