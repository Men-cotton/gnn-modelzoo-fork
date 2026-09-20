import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GraphSAGE as PyGGraphSAGE

from .cagnet_model import CagnetSAGE


class GraphSAGEWrapper(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        *,
        aggregator="mean",
    ):
        super().__init__()
        self.gnn = PyGGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Backbone outputs hidden_dim
            num_layers=num_layers,
            dropout=dropout,
            act="relu",
            norm=None,
            jk=None,
            aggr=aggregator,
            normalize=False,
            root_weight=True,
            project=False,
            bias=True,
        )
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout_p = dropout

    def forward(self, x, edge_index, batch_size=None, **kwargs):
        # PyG GraphSAGE doesn't require batch_size for standard forward
        x = self.gnn(x, edge_index)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.classifier(x)
        return x


class GCNWrapper(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        activation_hidden,
        activation_output,
        use_bias,
    ):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
            cached=False,
            normalize=True,
            bias=use_bias,
        )
        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
            cached=False,
            normalize=True,
            bias=use_bias,
        )
        self.dropout_p = dropout
        self.activation_hidden = _make_activation(activation_hidden)
        self.activation_output = _make_activation(activation_output)

    def forward(self, x, edge_index, batch_size=None, **kwargs):
        x = self.conv1(x, edge_index)
        x = self.activation_hidden(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.activation_output(x)
        return x


def _make_activation(name):
    normalized = str(name).lower()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "none":
        return torch.nn.Identity()
    raise ValueError(f"Unsupported PyG GNN activation '{name}'.")


def _architecture_type(architecture):
    return str(architecture.get("type", "graphsage")).lower()


def get_model(config, args=None, num_nodes=None):
    model_config = config["trainer"]["init"]["model"]
    architecture = model_config["architecture"]
    architecture_type = _architecture_type(architecture)

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

    if architecture_type in ("graphsage",):
        return _get_graphsage_model(
            architecture,
            use_cgnet,
            cagnet_rows,
            cagnet_cols,
            cagnet_rep,
            force_cagnet_flag,
            num_nodes,
        )

    if use_cgnet:
        raise ValueError(
            f"CAGNET arguments are only supported for GraphSAGE, not '{architecture_type}'."
        )

    if architecture_type in ("gcn",):
        return GCNWrapper(
            in_channels=architecture["n_feat"],
            hidden_channels=architecture["n_hid"],
            out_channels=architecture["n_class"],
            dropout=architecture["dropout_rate"],
            activation_hidden=architecture["activation_fn_hidden"],
            activation_output=architecture["activation_fn_output"],
            use_bias=architecture["use_bias"],
        )

    raise ValueError(f"Unsupported PyG GNN architecture '{architecture_type}'.")


def _get_graphsage_model(
    architecture,
    use_cgnet,
    cagnet_rows,
    cagnet_cols,
    cagnet_rep,
    force_cagnet_flag,
    num_nodes,
):
    aggregator = str(architecture.get("aggregator", "mean")).lower()
    if aggregator not in ("mean", "sum", "max"):
        raise ValueError(
            f"Unsupported GraphSAGE aggregator '{aggregator}'; "
            "choose mean, sum, or max."
        )
    if use_cgnet:
        if aggregator != "mean":
            raise ValueError("CAGNET GraphSAGE supports the mean aggregator only")
        # Always use CagnetSAGE (OFFSET-GNN baseline) if distributed or forced
        return CagnetSAGE(
            in_channels=architecture["n_feat"],
            hidden_channels=architecture["hidden_dim"],
            out_channels=architecture["n_class"],
            num_nodes=num_nodes,
            rows=cagnet_rows,
            cols=cagnet_cols,
            rep=cagnet_rep,
            dropout=architecture["dropout"],
            force_cagnet=force_cagnet_flag,
        )

    # Use native PyG GraphSAGE for efficiency in single-process / standard DDP.
    # The wrapper keeps the separate classifier head used by the CSZoo reference.
    return GraphSAGEWrapper(
        in_channels=architecture["n_feat"],
        hidden_channels=architecture["hidden_dim"],
        out_channels=architecture["n_class"],
        num_layers=architecture["num_layers"],
        dropout=architecture["dropout"],
        aggregator=aggregator,
    )
