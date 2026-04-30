from .gcn import GCN, GCNLayer
from .graphsage import GraphSAGE, GraphSAGELayer
from .registry import ArchitectureName, get_architecture_class

__all__ = [
    "ArchitectureName",
    "GCN",
    "GCNLayer",
    "GraphSAGE",
    "GraphSAGELayer",
    "get_architecture_class",
]
