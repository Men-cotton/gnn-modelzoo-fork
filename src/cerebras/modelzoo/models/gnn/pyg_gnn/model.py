import torch
from torch_geometric.nn.models import GraphSAGE
from .cagnet_model import CagnetSAGE

def get_model(config, args=None):
    m = config["trainer"]["init"]["model"]
    
    # Use CLI args if provided, otherwise fallback to config or defaults
    if args is not None and args.use_cagnet:
        use_cagnet = True
        cagnet_rows = args.cagnet_rows
        cagnet_cols = args.cagnet_cols
        cagnet_rep = args.cagnet_rep
    else:
        use_cagnet = m.get("use_cagnet", False)
        cagnet_rows = m.get("cagnet_rows", 1)
        cagnet_cols = m.get("cagnet_cols", 1)
        cagnet_rep = m.get("cagnet_rep", 1)

    if use_cagnet:
        # CAGNET-optimized model
        
        # CAGNET requires total number of nodes to distinguish full-batch
        # We might need to pass num_nodes in config or infer it. 
        # For now, let's assume it's passed or we default to 0 and handle it.
        # But wait, num_nodes is used in CagnetSAGE to check is_full_batch.
        # We should try to get it.
        num_nodes = m.get("num_nodes", 0)
        
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
    else:
        # Standard PyG SAGE
        model = GraphSAGE(
            in_channels=m["n_feat"],
            hidden_channels=m["graphsage_hidden_dim"],
            num_layers=m["graphsage_num_layers"],
            out_channels=m["n_class"],
            dropout=m["graphsage_dropout"],
            aggr=m["graphsage_aggregator"],
        )
    return model
