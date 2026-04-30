from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
import torch
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import AliasedPath
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.models.gnn.worker_validation import validate_num_workers

from .batches import FullGraphBatch, GraphSAGEBatch
from .samplers import (
    FullGraphDataProcessor,
    GraphSAGENeighborSamplerDataset,
    NeighborSamplingDataProcessor,
)
from .transforms import (
    normalize_adj_gcn,
)

logger = logging.getLogger(__name__)


class GNNDataProcessorConfig(DataConfig):
    data_processor: Literal["GNNDataProcessor"]
    dataset_name: Optional[str] = None
    dataset: Optional[str] = None
    dataset_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    data_dir: AliasedPath = "./data"

    sampling_mode: Literal["full_graph", "neighbor"] = "full_graph"
    fanouts: Optional[List[int]] = None
    caching_percent: Optional[float] = (
        None  # Percentage of nodes to cache on GPU (0.0 to 1.0)
    )
    sampler_seed: int = 0

    batch_size: int = Field(1)
    drop_last: bool = Field(True, validation_alias="drop_last_batch")
    num_workers: int = 0
    shuffle: bool = False
    prefetch_factor: Optional[int] = 10
    persistent_workers: bool = True
    pin_memory: bool = True

    split: Optional[Literal["train", "val", "valid", "test"]] = None
    adj_normalization: Optional[str] = (
        None  # Defaults to None (raw adjacency) unless specified (e.g. "gcn")
    )

    # Fake data support (full-graph only, retained for parity with legacy config)
    use_fake_data: bool = False
    fake_data_seed: Optional[int] = None
    fake_n_feat: int = 128
    fake_n_class: int = 7
    fake_num_nodes: int = 200
    pad_node_id: int = 0

    @model_validator(mode="before")
    @classmethod
    def _apply_dataset_profile(cls, values):
        if not isinstance(values, dict):
            return values

        dataset_key = values.get("dataset")
        profiles = values.get("dataset_profiles") or {}
        if dataset_key:
            dataset_key = str(dataset_key)
            normalized_key = dataset_key.lower()
            profile = None
            if isinstance(profiles, dict):
                profile = profiles.get(normalized_key) or profiles.get(dataset_key)
            if profile and isinstance(profile, dict):
                merged = dict(profile)
                merged.update(values)
                values = merged
            if not values.get("dataset_name"):
                values["dataset_name"] = dataset_key
            values["dataset"] = normalized_key
        return values

    @model_validator(mode="after")
    def _validate_dataset_name_present(self):
        if not self.dataset_name:
            raise ValueError(
                "GNNDataProcessorConfig requires 'dataset_name' to be specified "
                "either directly or via a dataset profile."
            )
        return self

    @field_validator("data_dir", mode="after")
    @classmethod
    def _normalize_data_dir(cls, value):
        if value is None:
            return value
        return os.path.abspath(value)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size_for_full_graph(cls, value, info):
        sampling_mode = info.data.get("sampling_mode", "full_graph")
        if (
            sampling_mode == "full_graph"
            and value != 1
            and not info.data.get("use_fake_data", False)
        ):
            logger.warning(
                "%s: batch_size is %s but will effectively be 1 for full-graph training.",
                cls.__name__,
                value,
            )
        return value

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers_not_exceed_cpu_cores(cls, value):
        return validate_num_workers(
            value,
            context=f"{cls.__name__}.num_workers",
        )

    @field_validator("split", mode="after")
    @classmethod
    def _normalize_split(cls, value):
        if value is None:
            return value
        return value


class GNNDataProcessor:
    """Facade that routes to the appropriate data pipeline per configuration."""

    def __init__(self, config: GNNDataProcessorConfig):
        if isinstance(config, dict):
            config = GNNDataProcessorConfig(**config)
        self.config = config

        self.float_dtype = cstorch.amp.get_floating_point_dtype()
        if self.config.sampling_mode == "full_graph":
            self.float_dtype = torch.float32
        self.label_dtype = torch.int32 if cstorch.use_cs() else torch.long
        self.current_split = getattr(self.config, "split", "train")

        self.adj_normalization_fn = None
        if self.config.adj_normalization == "gcn":
            self.adj_normalization_fn = normalize_adj_gcn
        elif (
            self.config.adj_normalization is not None
            and self.config.adj_normalization != "none"
        ):
            logger.warning(
                "Unknown adj_normalization '%s'; skipping normalization.",
                self.config.adj_normalization,
            )

        process_info = "Process"
        if cstorch.use_cs():
            try:
                process_info = f"Ordinal {dist.get_ordinal()}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to obtain ordinal: %s", exc)

        logger.info(
            "%s: Initializing GNNDataProcessor for dataset '%s' (split=%s, mode=%s).",
            process_info,
            self.config.dataset_name,
            self.current_split,
            self.config.sampling_mode,
        )

        if self.config.sampling_mode == "neighbor":
            if not self.config.fanouts:
                raise ValueError(
                    "Neighbor sampling requires 'fanouts' to be specified in the config."
                )
            self._processor = NeighborSamplingDataProcessor(
                dataset_name=self.config.dataset_name,
                data_dir=self.config.data_dir,
                current_split=self.current_split,
                float_dtype=self.float_dtype,
                label_dtype=self.label_dtype,
                adj_normalization_fn=self.adj_normalization_fn,
                fanouts=self.config.fanouts,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                sampler_seed=self.config.sampler_seed,
                num_workers=self.config.num_workers,
                pad_id=self.config.pad_node_id,
                caching_percent=self.config.caching_percent,
            )
        else:
            self._processor = FullGraphDataProcessor(
                dataset_name=self.config.dataset_name,
                data_dir=self.config.data_dir,
                current_split=self.current_split,
                float_dtype=self.float_dtype,
                label_dtype=self.label_dtype,
                adj_normalization_fn=self.adj_normalization_fn,
                drop_last=self.config.drop_last,
                num_workers=self.config.num_workers,
            )

    def create_dataloader(self) -> Union[DataLoader, SampleGenerator]:
        if self.config.use_fake_data:
            if not isinstance(self._processor, FullGraphDataProcessor):
                raise ValueError(
                    "Fake data generation is only supported for full-graph sampling."
                )
            return self._processor.create_fake_dataloader(
                num_nodes=self.config.fake_num_nodes,
                n_feat=self.config.fake_n_feat,
                n_class=self.config.fake_n_class,
                seed=self.config.fake_data_seed,
            )
        return self._processor.create_dataloader()


__all__ = [
    "GNNDataProcessor",
    "GNNDataProcessorConfig",
    "FullGraphBatch",
    "GraphSAGEBatch",
    "GraphSAGENeighborSamplerDataset",
    "normalize_adj_gcn",
]
