"""Opt-in native GPU optimizations; the SDK uses the unchanged GraphSAGE path."""

import time

import numpy as np
import torch

from .architectures.graphsage import GraphSAGE


class TopologyCache:
    """Sampled IDs/masks per node and hop, never features or learned states.

    Valid while graph, seed and fanouts are unchanged. Rebuild after changing
    topology. Target order, labels and features remain live on every input pass.
    The explicit byte bound avoids allocating a whole-graph table accidentally.
    """

    def __init__(self, dataset, max_bytes=256 * 1024**2):
        self.seed = dataset.seed
        self.fanouts = tuple(dataset.fanouts)
        self.num_nodes = dataset.num_nodes
        self.pad_id = dataset.pad_id
        self.nbytes = self.num_nodes * sum(self.fanouts) * 9
        if max_bytes <= 0 or self.nbytes > max_bytes:
            raise ValueError(
                f"Topology cache requires {self.nbytes} bytes; limit is {max_bytes}"
            )
        start = time.perf_counter()
        self.tables = []
        # Bounded scratch space, even when a full table fits the configured cap.
        for hop, fanout in enumerate(self.fanouts):
            ids = np.empty((self.num_nodes, fanout), dtype=np.int64)
            masks = np.empty((self.num_nodes, fanout), dtype=bool)
            for offset in range(0, self.num_nodes, 4096):
                nodes = np.arange(offset, min(offset + 4096, self.num_nodes))
                (
                    ids[offset : offset + len(nodes)],
                    masks[offset : offset + len(nodes)],
                ) = dataset._sample_neighbors(
                    nodes, np.ones(len(nodes), dtype=bool), fanout, hop
                )
            self.tables.append((ids, masks))
        self.build_seconds = time.perf_counter() - start

    def sample(self, dataset, nodes, valid, fanout, hop):
        if (
            dataset.seed != self.seed
            or tuple(dataset.fanouts) != self.fanouts
            or dataset.num_nodes != self.num_nodes
            or dataset.pad_id != self.pad_id
            or self.fanouts[hop] != fanout
        ):
            raise ValueError("Topology cache configuration changed; rebuild the loader")
        # Invalid parents may carry arbitrary IDs; never use those IDs as indices.
        safe = np.where(valid, nodes, 0)
        ids, masks = self.tables[hop]
        out_ids, out_masks = ids[safe].copy(), masks[safe].copy()
        out_ids[~valid] = self.pad_id
        out_masks[~valid] = False
        return out_ids, out_masks


def enable_topology_cache(dataset, max_bytes=256 * 1024**2):
    from .data_processing.samplers.neighbor_tree import GraphSAGENeighborSamplerDataset

    if not isinstance(dataset, GraphSAGENeighborSamplerDataset):
        raise ValueError(
            "Topology cache requires the ordinary neighbor sampler, not static replay"
        )
    if getattr(dataset, "_topology_cache", None) is not None:
        raise ValueError("Topology cache is already enabled")
    cache = TopologyCache(dataset, max_bytes)
    dataset._topology_cache = cache
    return cache


class DepthFusedGraphSAGE(GraphSAGE):
    """Batch each layer's linear maps across depths, preserving dropout calls.

    Floating point GEMM/reduction order can differ. Nonlinearities, dropout and
    parent masking retain the original deepest-to-shallowest call order/shape.
    No aggregation is moved across learned transforms or dropout.
    """

    def forward(self, batch):
        if (
            len(batch.node_features) != self.num_layers + 1
            or len(batch.node_masks) != self.num_layers + 1
            or len(batch.neighbor_masks) != self.num_layers
        ):
            raise ValueError("GraphSAGE batch depth mismatch")
        masks = [m.reshape(-1, 1) for m in batch.node_masks]
        neighbors = [m.reshape(-1, m.shape[-1]) for m in batch.neighbor_masks]
        states = [
            x.reshape(-1, x.shape[-1]) * m.to(x.dtype)
            for x, m in zip(batch.node_features, masks)
        ]
        for layer_index, layer in enumerate(self.layers):
            depths = list(range(self.num_layers - layer_index - 1, -1, -1))
            own = [states[d] for d in depths]
            means = [
                self._aggregate_neighbors(states[d + 1], neighbors[d]) for d in depths
            ]
            sizes = [x.shape[0] for x in own]
            combined = layer.self_linear(torch.cat(own, dim=0)) + layer.neighbor_linear(
                torch.cat(means, dim=0)
            )
            next_states = list(states)
            for depth, values in zip(depths, combined.split(sizes)):
                updated = layer.dropout(layer.activation(values))
                next_states[depth] = updated * masks[depth].to(updated.dtype)
            states = next_states
        return self.classifier(self.final_dropout(states[0]))


def fuse_depths(model):
    """Reuse the same parameters/state keys; choose this before compilation."""
    if type(model) is not GraphSAGE:
        raise ValueError("Depth fusion requires a vanilla GraphSAGE instance")
    model.__class__ = DepthFusedGraphSAGE
    return model
