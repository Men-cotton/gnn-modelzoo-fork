from __future__ import annotations

from typing import Dict, Literal, Type

import torch.nn as nn

from .gcn import GCN
from .graphsage import GraphSAGE

ArchitectureName = Literal["GCN", "GraphSAGE"]


_ARCHITECTURE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
}


def get_architecture_class(name: str) -> Type[nn.Module]:
    try:
        return _ARCHITECTURE_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported core architecture '{name}'.") from exc


__all__ = ["ArchitectureName", "get_architecture_class"]
