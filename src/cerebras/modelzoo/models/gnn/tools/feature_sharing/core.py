"""Portable fixed-shape inputs; SDK sample axis is distinct from logical roots."""

import hashlib
import inspect
import json
from pathlib import Path

import torch

from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.task.loss import masked_classification_loss


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    return tuple(map_tensors(v, fn) for v in value)


def restore(table, refs, logical_batch_size):
    return tuple(
        torch.gather(table, 1, r.long().expand(-1, -1, table.shape[-1])).reshape(
            logical_batch_size, -1, table.shape[-1]
        )
        for r in refs
    )


class SharedTreeModel(torch.nn.Module):
    def __init__(self, model_config, logical_batch_size, mode):
        super().__init__()
        self.sage = GraphSAGE(**model_config)
        self.logical_batch_size = logical_batch_size
        self.mode = mode

    def forward(self, batch):
        # Every streamed tensor has leading size 1. A logical tree is atomic
        # for this admission pilot; dictionaries never cross SDK samples.
        if self.mode == "dictionary":
            features = restore(batch["table"], batch["refs"], self.logical_batch_size)
        else:
            features = tuple(f[0] for f in batch["node_features"])
        payload = {
            k: map_tensors(batch[k], lambda t: t[0])
            for k in ("node_masks", "neighbor_masks", "labels", "target_mask")
        }
        payload["node_features"] = tuple(f.float() for f in features)
        logits = self.sage(GraphSAGEBatch.from_payload(payload))
        loss = masked_classification_loss(
            logits, payload["labels"], payload["target_mask"]
        )
        return loss, logits


class FrozenInputs(torch.utils.data.Dataset):
    """Small immutable window, repeated explicitly by DataExecutor if requested."""

    def __init__(self, input_dir, mode):
        self.root = Path(input_dir)
        self.metadata = json.loads((self.root / "input.json").read_text())
        native = Path(inspect.getfile(GraphSAGE)).resolve().parents[1]
        for name, digest in self.metadata["native_sha256"].items():
            if file_hash(native / name) != digest:
                raise ValueError(f"Native implementation checksum mismatch: {name}")
        self.mode = mode
        if mode not in ("dense", "dictionary"):
            raise ValueError("unknown input representation")
        for name, digest in self.metadata["files"].items():
            if file_hash(self.root / name) != digest:
                raise ValueError(f"Input checksum mismatch: {name}")
        self.samples = [
            torch.load(self.root / r["file"], weights_only=True, map_location="cpu")
            for r in self.metadata["batches"]
        ]
        self.capacity = self.metadata["capacity"]
        for sample in self.samples:
            if sample["table"].shape[0] > self.capacity:
                raise ValueError("dictionary overflow; no truncation or resampling")
            if not torch.equal(
                sample["table"][0], torch.zeros_like(sample["table"][0])
            ):
                raise ValueError("dictionary row zero must be zero")
            for refs in sample["refs"]:
                if refs.min() < 0 or refs.max() >= sample["table"].shape[0]:
                    raise ValueError("out-of-range dictionary reference")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.samples[index]
        batch = {
            k: map_tensors(sample[k], lambda t: t.unsqueeze(0))
            for k in (
                "node_masks",
                "neighbor_masks",
                "labels",
                "target_mask",
                "sample_id",
            )
        }
        refs = tuple(r.reshape(1, -1, 1) for r in sample["refs"])
        if self.mode == "dictionary":
            table = torch.zeros(
                1, self.capacity, sample["table"].shape[-1], dtype=sample["table"].dtype
            )
            table[0, : sample["table"].shape[0]] = sample["table"]
            batch.update(table=table, refs=refs)
        else:
            features = restore(
                sample["table"].unsqueeze(0), refs, self.metadata["logical_batch_size"]
            )
            batch["node_features"] = tuple(t.unsqueeze(0) for t in features)
        return batch


def input_fn(input_dir, mode):
    import cerebras.pytorch as cstorch

    if cstorch.distributed.is_streamer() and cstorch.distributed.num_streamers() != 1:
        raise ValueError("feature-sharing pilot requires exactly one input streamer")
    return torch.utils.data.DataLoader(
        FrozenInputs(input_dir, mode), batch_size=None, shuffle=False, num_workers=0
    )
