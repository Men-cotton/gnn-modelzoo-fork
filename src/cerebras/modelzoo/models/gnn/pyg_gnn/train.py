import os
import time
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.distributed as dist
from cerebras.pytorch.utils.tracker import RateTracker
from cerebras.modelzoo.models.gnn.pyg_gnn.eval import evaluate

def train_model(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    loaders: Tuple[Any, Any],
    data: Any,              # Restored to maintain API compatibility
    split_idx: Any,         # Restored to maintain API compatibility
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    cache: Optional[Any] = None
) -> None:
    """
    Executes the training loop with detailed granular profiling for HPC analysis.
    Arguments 'data' and 'split_idx' are retained for compatibility with the caller
    but are not used in this optimized loop.
    """
    if device.type != "cuda":
        raise RuntimeError("CUDA device required for precise timing measurements.")

    # --- Configuration Setup ---
    train_cfg = cfg["trainer"]["init"]
    loop_cfg = train_cfg["loop"]
    model_cfg = train_cfg["model"]
    
    model_dir = train_cfg["model_dir"]
    log_steps = train_cfg["logging"]["log_steps"]
    
    max_steps = int(loop_cfg["max_steps"])
    steps_per_epoch = int(loop_cfg["steps_per_epoch"])
    eval_frequency = int(loop_cfg.get("eval_frequency", steps_per_epoch)) # Default to steps_per_epoch if not set
    grad_accum_steps = int(loop_cfg.get("grad_accum_steps", 1))
    
    # --- Optimizer & AMP ---
    opt_conf = train_cfg["optimizer"]["AdamW"]
    optimizer = AdamW(
        model.parameters(),
        lr=opt_conf["learning_rate"],
        weight_decay=opt_conf["weight_decay"],
    )

    use_amp = bool(model_cfg.get("to_float16", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    disable_log_softmax = model_cfg.get("disable_log_softmax", False)
    compute_eval_metrics = bool(model_cfg.get("compute_eval_metrics", True))

    # --- Profiling Setup ---
    # Pre-allocate CUDA events to avoid allocation overhead during the loop
    profile_stages = ["h2d_struc", "h2d_fetch", "fwd", "bwd", "opt"]
    event_pool = [
        {stage: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) 
         for stage in profile_stages}
        for _ in range(log_steps)
    ]
    
    # Metrics aggregators
    metrics = {
        "prep_cpu": 0.0, "h2d_struc": 0.0, "h2d_fetch": 0.0, 
        "fwd": 0.0, "bwd": 0.0, "opt": 0.0, "gpu_total": 0.0
    }
    events_buffer = []  # Stores (cpu_prep_time, event_dict)
    
    # --- Training State ---
    train_loader, val_loader = loaders
    train_iter = iter(train_loader)
    
    model.train()
    running_loss_tensor = torch.zeros(1, device=device)
    step = 0
    
    # Rate tracker for throughput (samples/sec and edges/sec)
    rate_tracker = RateTracker()
    edge_rate_tracker = RateTracker()
    WARMUP_STEPS = 10

    # --- Helper: Flush Profiler Buffer ---
    def flush_profiler_buffer():
        """Synchronizes CUDA and aggregates timing data from the buffer."""
        torch.cuda.synchronize()
        nonlocal metrics
        
        for t_prep, evs in events_buffer:
            # Calculate elapsed times in ms
            t_h2d_struc = evs["h2d_struc"][0].elapsed_time(evs["h2d_struc"][1])
            t_h2d_fetch = evs["h2d_fetch"][0].elapsed_time(evs["h2d_fetch"][1])
            t_fwd = evs["fwd"][0].elapsed_time(evs["fwd"][1])
            t_bwd = evs["bwd"][0].elapsed_time(evs["bwd"][1])
            t_opt = evs["opt"][0].elapsed_time(evs["opt"][1])
            
            metrics["prep_cpu"] += t_prep * 1000.0  # sec to ms
            metrics["h2d_struc"] += t_h2d_struc
            metrics["h2d_fetch"] += t_h2d_fetch
            metrics["fwd"] += t_fwd
            metrics["bwd"] += t_bwd
            metrics["opt"] += t_opt
            metrics["gpu_total"] += (t_h2d_struc + t_h2d_fetch + t_fwd + t_bwd + t_opt)
        
        events_buffer.clear()

    # --- Sync Workers ---
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    
    total_wall_start = time.perf_counter()

    # --- Main Loop ---
    # Currently, not consider pipeline overlap
    while step < max_steps:
        # Select pre-allocated events for this step
        ev_current = event_pool[step % log_steps]

        # 1. Data Preparation (CPU)
        t_prep_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        t_prep = time.perf_counter() - t_prep_start

        # 2. Host-to-Device Transfer (Structure)
        ev_current["h2d_struc"][0].record()
        # already dropped batch.x in graphsage_pyg.py
        batch = batch.to(device, non_blocking=True) # pin_memory is enabled
        ev_current["h2d_struc"][1].record()

        # 3. Host-to-Device Transfer (Feature Fetch)
        ev_current["h2d_fetch"][0].record()
        if cache is not None:
            batch.x = cache.fetch(batch.n_id)
        ev_current["h2d_fetch"][1].record()

        # 4. Forward Pass
        ev_current["fwd"][0].record()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
            logits = logits[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            
            if not disable_log_softmax:
                logits = F.log_softmax(logits, dim=-1)
                loss = F.nll_loss(logits, y)
            else:
                loss = F.cross_entropy(logits, y)
        ev_current["fwd"][1].record()

        # 5. Backward Pass
        ev_current["bwd"][0].record()
        scaler.scale(loss / grad_accum_steps).backward()
        ev_current["bwd"][1].record()

        # 6. Optimization (amortized time)
        ev_current["opt"][0].record()
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        ev_current["opt"][1].record()

        # Store for delayed processing
        events_buffer.append((t_prep, ev_current))
        
        running_loss_tensor += loss.detach()
        rate_tracker.add(batch.batch_size)
        if hasattr(batch, 'num_edges'):
            edge_rate_tracker.add(batch.num_edges)
        step += 1

        # Reset profiler after warmup to exclude compilation/allocation overhead
        if step == WARMUP_STEPS:
            rate_tracker.reset()
            edge_rate_tracker.reset()
            # Reset accumulated metrics
            metrics = {k: 0.0 for k in metrics}
            events_buffer.clear() # Discard warmup events

        # --- Logging ---
        if step % log_steps == 0:
            flush_profiler_buffer()
            
            if dist.is_initialized():
                 dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
                 world_avg_loss = running_loss_tensor.item() / (world_size * log_steps)
            else:
                 world_avg_loss = running_loss_tensor.item() / log_steps
            
            # Reset

            if rank == 0:
                wall_time = time.perf_counter() - total_wall_start
                # Calculate denominator considering warmup
                denom = max(1, step - WARMUP_STEPS if step > WARMUP_STEPS else step)

                avg_m = {k: v / denom for k, v in metrics.items()}
                avg_load = avg_m["prep_cpu"] + avg_m["h2d_struc"] + avg_m["h2d_fetch"]

                print(f"[Step={step:04d}] Wall={wall_time:.4f}s | Loss={world_avg_loss:.4f}")
                print(f"[Profile] Avg ms/step | "
                      f"Load: {avg_load:.3f} (Prep: {avg_m['prep_cpu']:.3f}, Struc: {avg_m['h2d_struc']:.3f}, Fetch: {avg_m['h2d_fetch']:.3f}) | "
                      f"Fwd: {avg_m['fwd']:.3f} | Bwd: {avg_m['bwd']:.3f} | Opt: {avg_m['opt']:.3f} | "
                      f"GPU_Tot: {avg_m['gpu_total']:.3f}")
                print(f"[Throughput] Samples: {rate_tracker.global_rate():.2f} samples/s ({rate_tracker.rate():.2f}) | "
                      f"Edges: {edge_rate_tracker.global_rate():.2f} edges/s ({edge_rate_tracker.rate():.2f})")
            
            running_loss_tensor = torch.zeros(1, device=device)

        # --- Evaluation ---
        if compute_eval_metrics and step % eval_frequency == 0:
            # Note: evaluate is likely synchronous
            val_acc = evaluate(model, val_loader, device, cache=cache)
            model.train()
            
            if rank == 0:
                wall_time = time.perf_counter() - total_wall_start
                print(f"[Eval] Step={step:04d}, Wall={wall_time:.4f}s, Val_Acc={val_acc:.4f}")

    # --- Final Processing ---
    flush_profiler_buffer()
    
    if rank == 0:
        denom = max(1, step - WARMUP_STEPS if step > WARMUP_STEPS else step)
        avg_m = {k: v / denom for k, v in metrics.items()}
        
        print("-" * 60)
        print(f"Training Completed. Total Steps: {step} (Active: {denom})")
        print(f"Avg Breakdown (ms): Load={avg_m['prep_cpu']+avg_m['h2d_struc']+avg_m['h2d_fetch']:.3f} "
              f"[Prep:{avg_m['prep_cpu']:.3f}, Struc:{avg_m['h2d_struc']:.3f}, Fetch:{avg_m['h2d_fetch']:.3f}], "
              f"Fwd={avg_m['fwd']:.3f}, Bwd={avg_m['bwd']:.3f}, Opt={avg_m['opt']:.3f}")
        print("-" * 60)

        # Save Checkpoint
        ckpt_path = os.path.join(model_dir, "last.pt")
        torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
