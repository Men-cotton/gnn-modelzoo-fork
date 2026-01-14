import time
import torch
from cerebras.modelzoo.trainer.callbacks import Callback

class ComputeTimeCallback(Callback):
    def __init__(self):
        self.total_compute_time = 0.0
        self.start_time = None

    def on_fit_start(self, trainer, *args, **kwargs):
        """Reset state at the beginning of training."""
        self.total_compute_time = 0.0

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if not self._is_gnn_model(model):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if not self._is_gnn_model(model):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.total_compute_time += duration
            
            # Log to standard metrics (e.g. TensorBoard)
            trainer.log_metrics(compute_time=self.total_compute_time)

    def on_validate_end(self, trainer, model, loop):
        """Log compute time at the end of validation."""
        if not self._is_gnn_model(model):
            return
        trainer.logger.info(f"Step={trainer.global_step} compute_time={self.total_compute_time:.4f}s")

    def _is_gnn_model(self, model):
        # Loose check to avoid circular imports
        name = model.__class__.__name__
        return "GNN" in name or "GraphSAGE" in name
