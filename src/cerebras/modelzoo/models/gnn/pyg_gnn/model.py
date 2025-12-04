import torch
from torch_geometric.nn.models import GraphSAGE

def get_model(config):
    m = config["trainer"]["init"]["model"]
    model = GraphSAGE(
        in_channels=m["n_feat"],
        hidden_channels=m["graphsage_hidden_dim"],
        num_layers=m["graphsage_num_layers"],
        out_channels=m["n_class"],
        dropout=m["graphsage_dropout"],
        aggr=m["graphsage_aggregator"],
    )
    return model
