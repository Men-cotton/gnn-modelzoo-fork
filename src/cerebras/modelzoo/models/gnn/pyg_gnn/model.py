import torch
from torch_geometric.nn.models import GraphSAGE
from .cagnet_model import CagnetSAGE

def get_model(config, args=None, num_nodes=None):
    m = config["trainer"]["init"]["model"]
    
    # Defaults (1x1 CAGNET)
    cagnet_rows = 1
    cagnet_cols = 1
    cagnet_rep = 1
    force_cagnet_flag = False

    if args is not None:
        cagnet_rows = args.cagnet_rows
        cagnet_cols = args.cagnet_cols
        cagnet_rep = args.cagnet_rep
        force_cagnet_flag = getattr(args, "force_cagnet", False)
    
    # Determine if we should use CagnetSAGE or native GraphSAGE
    use_cgnet = False
    if cagnet_rows > 1 or cagnet_cols > 1 or cagnet_rep > 1:
        use_cgnet = True
    if force_cagnet_flag:
        use_cgnet = True

    if use_cgnet:
        # Always use CagnetSAGE (OFFSET-GNN baseline) if distributed or forced
        model = CagnetSAGE(
            in_channels=m["n_feat"],
            hidden_channels=m["graphsage_hidden_dim"],
            out_channels=m["n_class"],
            num_nodes=num_nodes,
            rows=cagnet_rows,
            cols=cagnet_cols,
            rep=cagnet_rep,
            dropout=m["graphsage_dropout"],
            force_cagnet=force_cagnet_flag
        )
    else:
        # Use native PyG GraphSAGE for efficiency in single-process / standard DDP
        model = GraphSAGE(
            in_channels=m["n_feat"],
            hidden_channels=m["graphsage_hidden_dim"],
            out_channels=m["n_class"],
            num_layers=2,
            dropout=m["graphsage_dropout"],
            act="relu",
            norm=None, # CagnetSAGE didn't use BN/LN, just row norm on adj (handled by SAGEConv)
            jk=None
        )

    return model
