"""Fixed-shape history GraphSAGE using the native GraphSAGE parameters and loss.

Mean aggregation permits sending the boundary mean rather than F embeddings.
Current in-batch neighbors retain their autograd paths. Historical boundary
neighbors are constants. A single sampled graph is shared across depths, so
the recursive baseline and a fresh history have identical forward semantics.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
from cerebras.modelzoo.models.gnn.task.loss import masked_classification_loss
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch


class HistorySAGE(GraphSAGE):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super().__init__(
            input_dim, hidden_dim, num_layers, dropout, "mean", num_classes
        )

    def forward(self, batch):
        if "node_features" in batch and "node_means" not in batch:
            payload = GraphSAGEBatch(
                node_features=batch["node_features"],
                node_masks=tuple(
                    torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
                    for x in batch["node_features"]
                ),
                neighbor_masks=batch["neighbor_masks"],
                labels=batch["labels"],
                target_mask=batch["valid"],
            )
            return super().forward(payload), ()
        if "node_features" in batch:
            states = list(batch["node_features"])
            for i, layer in enumerate(self.layers):
                updated = [None] * (len(self.layers) - i)
                for depth in range(len(self.layers) - i - 1, -1, -1):
                    if i == 0 and "node_means" in batch:
                        mean = batch["node_means"][depth]
                    else:
                        mean = self._aggregate_neighbors(
                            states[depth + 1], batch["neighbor_masks"][depth]
                        )
                    updated[depth] = layer(states[depth], mean)
                states = updated
            return self.classifier(self.final_dropout(states[0])), ()
        x = batch["x"]
        histories = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                mean = batch["x_mean"]
            else:
                local = x.index_select(0, batch["local_index"].flatten().long())
                local = local.reshape(*batch["local_index"].shape, x.shape[-1])
                mean = (local * batch["local_mask"].unsqueeze(-1)).sum(1)
                mean = mean * batch["inv_degree"] + batch["boundary_mean"][i - 1]
            x = layer(x, mean) * batch["valid"].unsqueeze(-1)
            if i != len(self.layers) - 1:
                histories.append(x)
        return self.classifier(self.final_dropout(x)), tuple(histories)

    def tree(self, features, neighbors, mask, roots):
        """Exact recursive expansion with gradients through every occurrence."""
        batch = prepare_tree(
            features,
            neighbors,
            mask,
            torch.zeros(len(features), dtype=torch.long, device=features.device),
            roots,
            self.num_layers,
        )
        return self(batch)[0]


def masked_loss(logits, labels, valid):
    return masked_classification_loss(logits, labels, valid, disable_log_softmax=True)


@torch.no_grad()
def neighbor_mean(values, neighbors, mask, chunk=2048):
    result = torch.empty(
        (len(values), values.shape[1]), dtype=values.dtype, device=values.device
    )
    for start in range(0, len(values), chunk):
        n = neighbors[start : start + chunk]
        m = mask[start : start + chunk]
        result[start : start + chunk] = (values[n] * m.unsqueeze(-1)).sum(1) / m.sum(
            1
        ).clamp_min(1).unsqueeze(-1)
    return result


@torch.no_grad()
def refresh(model, features, neighbors, mask, chunk=2048):
    """Exact, dropout-free layerwise refresh at one weight version, on CPU/GPU."""
    training = model.training
    model.eval()
    histories = []
    x = features
    for layer in model.layers[:-1]:
        mean = neighbor_mean(x, neighbors, mask, chunk)
        x = torch.cat(
            [
                layer(x[s : s + chunk], mean[s : s + chunk])
                for s in range(0, len(x), chunk)
            ]
        )
        histories.append(x.detach())
    model.train(training)
    return histories


@torch.no_grad()
def infer(model, features, neighbors, mask, chunk=2048):
    training = model.training
    model.eval()
    x = features
    for layer in model.layers:
        mean = neighbor_mean(x, neighbors, mask, chunk)
        x = torch.cat(
            [
                layer(x[s : s + chunk], mean[s : s + chunk])
                for s in range(0, len(x), chunk)
            ]
        )
    logits = torch.cat(
        [model.classifier(x[s : s + chunk]) for s in range(0, len(x), chunk)]
    )
    model.train(training)
    return logits


class HistoryStore:
    """Single-owner host table. No DataLoader worker may mutate this object.

    A frozen epoch snapshot may cache global neighbor means. Online writes
    invalidate those means; in that mode gather boundary rows for each batch.
    Versions identify pre-forward weights by completed optimizer steps. Duplicate-ID/out-of-order writes
    are rejected before mutation. Padding never writes node zero.
    """

    def __init__(self, histories, step=0):
        self.values = [x.detach().cpu().clone() for x in histories]
        self.version = (
            torch.full((len(histories[0]),), step, dtype=torch.int64)
            if histories
            else None
        )
        self.means = None

    def cache_means(self, neighbors, mask):
        self.means = [neighbor_mean(x, neighbors, mask) for x in self.values]

    def write(self, roots, outputs, step):
        if len(roots) != len(torch.unique(roots)):
            raise ValueError("Duplicate history write IDs")
        if len(outputs) != len(self.values):
            raise ValueError("History layer count mismatch")
        if self.version is not None and (self.version[roots] > step).any():
            raise ValueError("Out-of-order history update")
        outputs = [x.detach().cpu() for x in outputs]
        for old, new in zip(self.values, outputs):
            if new.shape != (len(roots), old.shape[1]) or not torch.isfinite(new).all():
                raise ValueError("Invalid history shape/value")
        for old, new in zip(self.values, outputs):
            old[roots] = new
        if self.version is not None:
            self.version[roots] = step
        self.means = None


def prepare(features, x_mean, neighbors, mask, labels, roots, store, batch_size=None):
    """Host boundary preaggregation; all returned tensors have fixed shapes."""
    if features.device.type != "cpu":
        raise ValueError("prepare is the host input pipeline")
    roots = roots.cpu()
    count = len(roots)
    size = batch_size or count
    if count == 0 or count > size or len(torch.unique(roots)) != count:
        raise ValueError("Expected nonempty unique roots fitting the batch")
    padded = F.pad(roots, (0, size - count))
    valid = torch.arange(size) < count
    n = neighbors[padded]
    m = mask[padded] & valid.unsqueeze(-1)
    # Lookup only against real roots: padded node zero is never an in-batch node.
    sorted_roots, order = roots.sort()
    pos = torch.searchsorted(sorted_roots, n).clamp_max(count - 1)
    local_mask = (sorted_roots[pos] == n) & m
    local_index = order[pos].to(torch.int32)
    inv_degree = m.sum(1).clamp_min(1).reciprocal().unsqueeze(-1)
    boundary = []
    for i, history in enumerate(store.values):
        if store.means is None:
            value = (history[n] * (m & ~local_mask).unsqueeze(-1)).sum(1) * inv_degree
        else:
            local_old = history[padded][local_index.long()]
            value = (
                store.means[i][padded]
                - (local_old * local_mask.unsqueeze(-1)).sum(1) * inv_degree
            )
            value = value * valid.unsqueeze(-1)
        boundary.append(value)
    return {
        "x": features[padded],
        "x_mean": x_mean[padded],
        "boundary_mean": tuple(boundary),
        "local_index": local_index,
        "local_mask": local_mask,
        "inv_degree": inv_degree,
        "valid": valid,
        "labels": labels[padded],
        "root_ids": padded,
    }


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(x) for x in value.values())
    return sum(tensor_bytes(x) for x in value)


def prepare_tree(features, neighbors, mask, labels, roots, layers, batch_size=None):
    size = batch_size or len(roots)
    if not 0 < len(roots) <= size:
        raise ValueError("Invalid tree batch size")
    padded = F.pad(roots, (0, size - len(roots)))
    ids = [padded]
    masks = []
    for _ in range(layers):
        masks.append(mask[ids[-1]])
        ids.append(neighbors[ids[-1]].flatten())
    return {
        "node_features": tuple(features[index] for index in ids),
        "neighbor_masks": tuple(masks),
        "labels": labels[padded],
        "valid": torch.arange(size) < len(roots),
        "root_ids": padded,
    }


def prepare_tree_mean(
    features, x_mean, neighbors, mask, labels, roots, layers, batch_size=None
):
    """Exact, no-history control: move ONLY raw-feature means to the host."""
    size = batch_size or len(roots)
    if not 0 < len(roots) <= size:
        raise ValueError("Invalid tree batch size")
    padded = F.pad(roots, (0, size - len(roots)))
    ids = [padded]
    masks = []
    for _ in range(layers - 1):
        masks.append(mask[ids[-1]])
        ids.append(neighbors[ids[-1]].flatten())
    return {
        "node_features": tuple(features[index] for index in ids),
        "node_means": tuple(x_mean[index] for index in ids),
        "neighbor_masks": tuple(masks),
        "labels": labels[padded],
        "valid": torch.arange(size) < len(roots),
        "root_ids": padded,
    }


def to_device(batch, device):
    return {
        k: tuple(x.to(device) for x in v) if isinstance(v, tuple) else v.to(device)
        for k, v in batch.items()
    }
