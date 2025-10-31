from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
import torch
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator

from .batches import GraphSAGEBatch
from .pipelines import (
    FullGraphDataProcessor,
    GraphSAGENeighborSamplerDataset,
    NeighborSamplingDataProcessor,
    normalize_adj_gcn,
)

logger = logging.getLogger(__name__)

_DATASET_NAME_ALIASES: Dict[str, str] = {
    "pubmed": "PubMed",
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "reddit": "Reddit",
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn_arxiv": "ogbn-arxiv",
    "ogbn-mag": "ogbn-mag",
    "ogbn_mag": "ogbn-mag",
    "ogbn-products": "ogbn-products",
    "ogbn_products": "ogbn-products",
    "ogbn-papers100m": "ogbn-papers100M",
    "ogbn_papers100m": "ogbn-papers100M",
    "mag240m": "MAG240M",
}


class GNNDataProcessorConfig(DataConfig):
    data_processor: Literal["GNNDataProcessor"]
    dataset_name: Optional[str] = None
    dataset: Optional[str] = None
    dataset_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    data_dir: str = "./data"

    sampling_mode: Literal["full_graph", "neighbor"] = "full_graph"
    fanouts: Optional[List[int]] = None
    sampler_seed: int = 0

    batch_size: int = Field(1)
    drop_last: bool = Field(True, validation_alias="drop_last_batch")
    num_workers: int = 0
    shuffle: bool = False

    split: Optional[Literal["train", "val", "test"]] = None
    adj_normalization: Optional[str] = "gcn"

    # Fake data support (full-graph only, retained for parity with legacy config)
    use_fake_data: bool = False
    fake_data_seed: Optional[int] = None
    fake_n_feat: int = 128
    fake_n_class: int = 7
    fake_num_nodes: int = 200
    pad_node_id: int = 0

    @staticmethod
    def _resolve_dataset_name(dataset_key: str) -> str:
        return _DATASET_NAME_ALIASES.get(dataset_key.lower(), dataset_key)

    @model_validator(mode="before")
    @classmethod
    def _apply_dataset_profile(cls, values):
        if not isinstance(values, dict):
            return values

        dataset_key = values.get("dataset")
        profiles = values.get("dataset_profiles") or {}
        if dataset_key:
            normalized_key = str(dataset_key).lower()
            profile = None
            if isinstance(profiles, dict):
                profile = profiles.get(normalized_key) or profiles.get(dataset_key)
            if profile and isinstance(profile, dict):
                merged = dict(profile)
                merged.update(values)
                values = merged
            if not values.get("dataset_name"):
                values["dataset_name"] = cls._resolve_dataset_name(normalized_key)
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
                fake_num_nodes=self.config.fake_num_nodes,
                fake_n_feat=self.config.fake_n_feat,
                fake_n_class=self.config.fake_n_class,
                fake_data_seed=self.config.fake_data_seed,
            )
        return self._processor.create_dataloader()


__all__ = [
    "GNNDataProcessor",
    "GNNDataProcessorConfig",
    "GraphSAGEBatch",
    "GraphSAGENeighborSamplerDataset",
    "normalize_adj_gcn",
]
