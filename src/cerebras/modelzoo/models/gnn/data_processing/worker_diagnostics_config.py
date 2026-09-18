"""Optional host-side observations for the GNN neighbor input pipeline."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerDiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    output_dir: Optional[str] = None
    max_batches: int = Field(16, ge=0, le=10000)
    snapshot_interval_seconds: float = Field(1.0, ge=0.1, allow_inf_nan=False)
    max_snapshots: int = Field(30, ge=0, le=1000)
    resource_monitor: bool = False
    resource_monitor_pss: bool = False

    @model_validator(mode="after")
    def require_output_directory(self):
        if self.enabled and (not self.output_dir or not self.output_dir.strip()):
            raise ValueError("enabled worker_diagnostics requires output_dir")
        return self
