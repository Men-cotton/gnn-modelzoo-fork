from __future__ import annotations

from typing import Dict, Literal, Type

import torch.nn as nn

from .gatv2 import GATv2
from .gcn import GCN
from .graphsage import GraphSAGE

ArchitectureName = Literal["GATv2", "GCN", "GraphSAGE"]


_ARCHITECTURE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gatv2": GATv2,
    "gcn": GCN,
    "graphsage": GraphSAGE,
}


def get_architecture_class(name: str) -> Type[nn.Module]:
    try:
        return _ARCHITECTURE_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported core architecture '{name}'.") from exc


__all__ = ["ArchitectureName", "get_architecture_class"]
