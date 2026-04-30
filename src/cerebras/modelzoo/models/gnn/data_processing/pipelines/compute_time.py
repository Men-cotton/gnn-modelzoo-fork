import cerebras.pytorch as cstorch
import torch
from cerebras.pytorch.utils.tracker import RateTracker
from cerebras.pytorch.utils.data.utils import infer_batch_size
from cerebras.modelzoo.trainer.callbacks import Callback
import time


class ComputeTimeCallback(Callback):
    """
    HPC Logger using precise 'before/after' hooks to isolate host dispatch latency.
    """

    def __init__(self, log_steps: int = 10, warmup_steps: int = 10):
        self.log_steps = log_steps
        self.warmup_steps = warmup_steps

        self.rate_tracker = RateTracker()

        self.metrics = {
            "load": 0.0,
            "host_submit_fwd": 0.0,
            "host_submit_bwd": 0.0,
            "host_submit_opt": 0.0,
            "iter_wall": 0.0,
        }

        # Timestamps
        self.total_wall_start = 0.0
        self.t_batch_end = 0.0
        self.t_batch_start = 0.0
        self.t_temp_start = 0.0  # Temporary storage for 'before' hook
        self.in_warmup = True
        self.pause_start_time = None

        # State tracking
        self._in_train_loop = False
        self.window_steps = 0

        # Compilation tracking
        self.is_compiled = False
        self.compile_start_time = 0.0

    def on_fit_start(self, trainer, *args, **kwargs):
        if hasattr(trainer, "logging") and hasattr(trainer.logging, "log_steps"):
            self.log_steps = trainer.logging.log_steps
        self.total_wall_start = time.perf_counter()
        self.t_batch_end = time.perf_counter()
        self.compile_start_time = time.perf_counter()

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self._in_train_loop = True
        if self.pause_start_time is not None:
            # We are resuming from a pause (likely validation)
            duration = time.perf_counter() - self.pause_start_time
            self.total_wall_start += duration
            self.t_batch_end += duration

            # Correct RateTracker by effectively subtracting the paused duration
            elapsed = self.rate_tracker.elapsed_seconds()
            if elapsed > duration:
                self.rate_tracker.reset_time(offset=(elapsed - duration))

            self.pause_start_time = None

    def on_train_end(self, trainer, model, loop, loop_idx):
        # Pause timer when exiting training loop (e.g. for validation)
        self._in_train_loop = False
        self.pause_start_time = time.perf_counter()

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        t_now = time.perf_counter()
        # "load" is the gap between the end of previous batch and start of current
        self.metrics["load"] += (t_now - self.t_batch_end) * 1000.0
        self.t_batch_start = t_now

    # --- Forward Wrapping ---
    def on_before_forward(self, trainer, model, batch, args, kwargs):
        if not self._in_train_loop:
            return
        self.t_temp_start = time.perf_counter()

    def on_after_forward(self, trainer, model, outputs, batch):
        if not self._in_train_loop:
            return
        duration = time.perf_counter() - self.t_temp_start
        self.metrics["host_submit_fwd"] += duration * 1000.0

    # --- Backward Wrapping ---
    def on_before_backward(self, trainer, model, outputs):
        if not self._in_train_loop:
            return
        self.t_temp_start = time.perf_counter()

    def on_after_backward(self, trainer, model, outputs):
        if not self._in_train_loop:
            return
        duration = time.perf_counter() - self.t_temp_start
        self.metrics["host_submit_bwd"] += duration * 1000.0

    # --- Optimizer Wrapping ---
    def on_before_optimizer_step(self, trainer, model, optimizer):
        if not self._in_train_loop:
            return
        self.t_temp_start = time.perf_counter()

    def on_after_optimizer_step(self, trainer, model, optimizer):
        if not self._in_train_loop:
            return
        duration = time.perf_counter() - self.t_temp_start
        self.metrics["host_submit_opt"] += duration * 1000.0

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        # Throughput tracking
        batch_size = infer_batch_size(batch)
        if batch_size is None:
            batch_size = 1

        self.rate_tracker.add(batch_size)

        # Step Wall Time and Residual
        t_now = time.perf_counter()
        iter_wall = (t_now - self.t_batch_start) * 1000.0

        self.metrics["iter_wall"] += iter_wall

        step = trainer.global_step
        self.window_steps += 1

        # Compilation Isolation (Step 1)
        # We treat the first completed step as the compilation step.
        if not self.is_compiled and step >= 1:
            compile_dur = time.perf_counter() - self.compile_start_time
            print(f"[Compilation] Step={step} finished. Duration={compile_dur:.2f}s")
            self.is_compiled = True

            # Reset trackers to avoid polluting steady-state metrics
            self.rate_tracker.reset()
            for k in self.metrics:
                self.metrics[k] = 0.0
            self.window_steps = 0
            self.total_wall_start = time.perf_counter()
            self.t_batch_end = time.perf_counter()
            return  # Skip standard logging for this outlier step

        # Warmup logic
        if step == self.warmup_steps:
            self.rate_tracker.reset()
            for k in self.metrics:
                self.metrics[k] = 0.0
            self.window_steps = 0
            self.in_warmup = False
            self.total_wall_start = time.perf_counter()
            self.t_batch_end = time.perf_counter()
            return  # Skip logging at the warmup boundary to avoid printing zero/reset metrics

        # Logging
        if not self.in_warmup and step % self.log_steps == 0:
            self._print_log(trainer, step, outputs)
            for k in self.metrics:
                self.metrics[k] = 0.0
            self.window_steps = 0

        self.t_batch_end = time.perf_counter()

    def _print_log(self, trainer, step, outputs):
        # Avoid materializing tensor values outside step closures on CS.
        loss_obj = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_obj = outputs["loss"]
        elif torch.is_tensor(outputs):
            loss_obj = outputs

        # Averages
        if self.window_steps == 0:
            return  # Avoid division by zero

        avg_load = self.metrics["load"] / self.window_steps
        avg_fwd = self.metrics["host_submit_fwd"] / self.window_steps
        avg_bwd = self.metrics["host_submit_bwd"] / self.window_steps
        avg_opt = self.metrics["host_submit_opt"] / self.window_steps
        avg_wall = self.metrics["iter_wall"] / self.window_steps
        avg_host = avg_fwd + avg_bwd + avg_opt
        # WSE "Device" Time (Residual)
        # We display Host Submit separately from Residual.
        avg_residual = avg_wall - avg_host

        # Detect measurement breakdown (negative residual)
        if avg_residual < -1e-3:
            print(f"[Warn] Residual became negative: {avg_residual:.3f} ms/step")

        wall_time = time.perf_counter() - self.total_wall_start

        # Capture rates now so they match this logging window.
        global_rate = self.rate_tracker.global_rate()
        local_rate = self.rate_tracker.rate()

        self._print_log_step_closure(
            trainer,
            step,
            loss_obj,
            wall_time,
            avg_load,
            avg_fwd,
            avg_bwd,
            avg_opt,
            avg_residual,
            avg_wall,
            global_rate,
            local_rate,
        )

    @cstorch.step_closure
    def _print_log_step_closure(
        self,
        trainer,
        step,
        loss_obj,
        wall_time,
        avg_load,
        avg_fwd,
        avg_bwd,
        avg_opt,
        avg_residual,
        avg_wall,
        global_rate,
        local_rate,
    ):
        # Materialize loss only inside a step closure for CS compatibility.
        loss_val = 0.0
        if torch.is_tensor(loss_obj):
            loss_val = loss_obj.item()
        elif loss_obj is not None:
            try:
                loss_val = float(loss_obj)
            except (TypeError, ValueError):
                loss_val = 0.0

        print(f"[Step={step:04d}] Wall={wall_time:.4f}s | Loss={loss_val:.4f}")
        # Updated Profile Log
        print(
            f"[Profile] Avg ms/step | "
            f"Load: {avg_load:.3f} | "
            f"Host_Submit(Fwd: {avg_fwd:.3f}, Bwd: {avg_bwd:.3f}, Opt: {avg_opt:.3f}) | "
            f"Residual(Dev): {avg_residual:.3f} | "
            f"Iter_Wall: {avg_wall:.3f}"
        )

        print(f"[Throughput] Samples: {global_rate:.2f} samples/s ({local_rate:.2f})")
