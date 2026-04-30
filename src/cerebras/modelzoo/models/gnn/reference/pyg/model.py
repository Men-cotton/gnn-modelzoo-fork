import torch
from torch_geometric.nn.models import GraphSAGE
import torch.nn.functional as F
from .cagnet_model import CagnetSAGE


class GraphSAGEWrapper(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.gnn = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Backbone outputs hidden_dim
            num_layers=num_layers,
            dropout=dropout,
            act="relu",
            norm=None,
            jk=None,
        )
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout_p = dropout

    def forward(self, x, edge_index, batch_size=None, **kwargs):
        # PyG GraphSAGE doesn't require batch_size for standard forward
        x = self.gnn(x, edge_index)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.classifier(x)
        return x


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
            force_cagnet=force_cagnet_flag,
        )
    else:
        # Use native PyG GraphSAGE for efficiency in single-process / standard DDP
        # We use a wrapper to ensure a separate classifier head, matching CSZoo reference architecture
        model = GraphSAGEWrapper(
            in_channels=m["n_feat"],
            hidden_channels=m["graphsage_hidden_dim"],
            out_channels=m["n_class"],
            num_layers=m["graphsage_num_layers"],
            dropout=m["graphsage_dropout"],
        )

    return model
