import torch
from torch_geometric.nn.models import GraphSAGE
from .cagnet_model import CagnetSAGE

def get_model(config, args=None, num_nodes=None):
    m = config["trainer"]["init"]["model"]
    
    # Defaults (1x1 CAGNET)
    cagnet_rows = 1
    cagnet_cols = 1
    cagnet_rep = 1

    if args is not None:
        cagnet_rows = args.cagnet_rows
        cagnet_cols = args.cagnet_cols
        cagnet_rep = args.cagnet_rep

    # Resolve num_nodes
    if num_nodes is None:
        num_nodes = m.get("num_nodes", 0)

    # Always use CagnetSAGE (OFFSET-GNN baseline)
    model = CagnetSAGE(
        in_channels=m["n_feat"],
        hidden_channels=m["graphsage_hidden_dim"],
        out_channels=m["n_class"],
        num_nodes=num_nodes,
        rows=cagnet_rows,
        cols=cagnet_cols,
        rep=cagnet_rep,
        dropout=m["graphsage_dropout"],
        force_cagnet=m.get("force_cagnet", False)
    )
    return model
