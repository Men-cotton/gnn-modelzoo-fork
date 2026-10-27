"""Local CPU feasibility probe; no production loader changes or CSX execution.

Host packs identical native sampled trees into fixed-size feature dictionaries
and int32 references. Device-style torch.gather reconstructs every padded slot.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
from benchmark_scripts.research.logs.local_factorial_sampler import FixedCyclicTreeProvider
from benchmark_scripts.research.logs.local_ogb_graph import load_local_ogb_graph


def pack(dataset, tree, group_size, capacity):
    nodes, masks, _ = tree
    batch_size, width = dataset.batch_size, dataset.features.shape[1]
    if batch_size % group_size:
        raise ValueError("group_size must divide the padded root batch")
    groups = batch_size // group_size
    tables = torch.zeros(groups, capacity, width, dtype=dataset.features.dtype)
    refs = [np.zeros((groups, a.size // groups), dtype=np.int32) for a in nodes]
    counts = []
    for group in range(groups):
        active_ids = [a.reshape(groups, -1)[group][m.reshape(groups, -1)[group]]
                      for a, m in zip(nodes, masks)]
        unique = np.unique(np.concatenate(active_ids))
        if unique.size + 1 > capacity:
            raise ValueError("dictionary capacity exceeded; never truncate or resample")
        counts.append(int(unique.size))
        tables[group, 1:unique.size+1] = dataset.features[torch.from_numpy(unique)]
        for depth, (a, m) in enumerate(zip(nodes, masks)):
            flags = m.reshape(groups, -1)[group]
            ids = a.reshape(groups, -1)[group][flags]
            refs[depth][group, flags] = np.searchsorted(unique, ids).astype(np.int32) + 1
    return tables, [torch.from_numpy(r) for r in refs], counts


def unpack(tables, refs, batch_size):
    return [torch.gather(tables, 1, r.long().unsqueeze(-1).expand(-1, -1, tables.shape[-1]))
            .reshape(batch_size, -1, tables.shape[-1]) for r in refs]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["arxiv", "products"], required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Existing neighbor-redundancy batches.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.modelzoo_root / "src"))
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import GraphSAGENeighborSamplerDataset
    from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
    from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
    import cerebras.pytorch as cstorch

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    graph_args = load_local_ogb_graph("ogbn-" + args.dataset, args.modelzoo_root /
                                    "src/cerebras/modelzoo/models/gnn/data/datasets")
    graph_meta = graph_args.pop("metadata")
    common = dict(**graph_args, fanouts=[15, 10, 5], shuffle=True, pad_id=0, seed=42)
    small = GraphSAGENeighborSamplerDataset(**common, batch_size=32)
    checks = []
    for batch_index in [0, len(small)-1]:
        roots = np.zeros(32, dtype=np.int64)
        active = np.zeros(32, dtype=bool)
        targets = small._ordered_targets[batch_index*32:(batch_index+1)*32]
        roots[:len(targets)], active[:len(targets)] = targets, True
        tree = small._sample_layers(roots, active)
        expected = small[batch_index]
        capacity = 8*916 + 1  # guaranteed bound, no observed-maximum assumption
        table, refs, counts = pack(small, tree, 8, capacity)
        restored = unpack(table, refs, 32)
        for actual, reference in zip(restored, expected["node_features"]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # The group dimension is separable in the reconstruction operator.
        split = [unpack(table[g:g+1], [r[g:g+1] for r in refs], 8) for g in range(4)]
        for depth, reference in enumerate(restored):
            torch.testing.assert_close(torch.cat([s[depth] for s in split]), reference, rtol=0, atol=0)
        model = GraphSAGE(input_dim=table.shape[-1], hidden_dim=16, num_layers=3,
                         dropout=.5, aggregator="mean", num_classes=3).train()
        def run(features):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1234)
            payload = dict(expected, node_features=[f.float() for f in features])
            logits = model(GraphSAGEBatch.from_payload(payload))
            valid_logits = logits[expected["target_mask"]]
            loss = valid_logits.square().mean()
            loss.backward()
            return logits.detach().clone(), loss.detach().clone(), [p.grad.clone() for p in model.parameters()]
        original, compressed = run(expected["node_features"]), run(restored)
        for a, b in zip(original[:2], compressed[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(original[2], compressed[2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        try:
            pack(small, tree, 8, 1)
        except ValueError:
            overflow_rejected = True
        else:
            raise AssertionError("overflow was not rejected")
        checks.append(dict(batch_index=batch_index, real_targets=len(targets),
                           group_unique_counts=counts, capacity=capacity,
                           features_exact=True, logits_loss_parameter_gradients_exact=True,
                           dropout_rng_replayed=True, group_reconstruction_separable=True,
                           overflow_rejected=overflow_rejected))
        print(json.dumps(checks[-1]), flush=True)

    # First full batch only: grouped capacity sensitivity, not a full-epoch bound.
    large = GraphSAGENeighborSamplerDataset(**common, batch_size=4096)
    nodes, masks, _ = FixedCyclicTreeProvider(large, seed=42).tree(0)
    valid = [a[m] for a, m in zip(nodes, masks)]
    signature = hashlib.sha256(np.asarray(np.concatenate(valid), dtype="<i8").tobytes()).hexdigest()
    baseline = args.baseline
    first = json.loads(baseline.read_text().splitlines()[0])
    assert signature == first["tree_ids_sha256"]
    sensitivity = []
    feature_bytes = graph_meta["feature_dim"] * 2
    dense_bytes = 4096*916*feature_bytes
    for group_size in (1, 16, 64, 256, 4096):
        groups = 4096 // group_size
        counts = [int(np.unique(np.concatenate([
            a.reshape(groups, -1)[g][m.reshape(groups, -1)[g]]
            for a, m in zip(nodes, masks)])).size) for g in range(groups)]
        observed_capacity = ((max(counts)+1+255)//256)*256
        guaranteed_capacity = min(graph_meta["num_nodes"], group_size*916)+1
        index_bytes = 4096*916*4
        sensitivity.append(dict(group_size=group_size, groups=groups, unique_counts=counts,
            observed_capacity_rounded_256=observed_capacity,
            guaranteed_capacity=guaranteed_capacity,
            observed_capacity_payload_bytes=groups*observed_capacity*feature_bytes+index_bytes,
            guaranteed_capacity_payload_bytes=groups*guaranteed_capacity*feature_bytes+index_bytes,
            dense_feature_bytes=dense_bytes, int32_reference_bytes=index_bytes))
    # Full graph capacity is a guarantee for any target order/sampling seed.
    guaranteed_bytes=(graph_meta["num_nodes"]+1)*feature_bytes + 4096*916*4
    files=[Path(__file__),Path(inspect.getfile(GraphSAGE)),Path(inspect.getfile(GraphSAGENeighborSamplerDataset)),
           Path(inspect.getfile(FixedCyclicTreeProvider)),Path(inspect.getfile(load_local_ogb_graph))]
    result=dict(status="complete", dataset=args.dataset, device="cpu", torch=torch.__version__,
                cerebras_pytorch=cstorch.__version__, graph=graph_meta, checks=checks,
                first_full_batch_tree_digest_matches_baseline=True,
                grouping_sensitivity_scope="first full seed42 batch only; capacities estimated from this batch are not guarantees",
                grouping_sensitivity=sensitivity, dense_feature_bytes=dense_bytes,
                full_graph_capacity_payload_bytes=guaranteed_bytes,
                full_graph_capacity_payload_ratio=guaranteed_bytes/dense_bytes,
                payload_scope="FP16 feature table plus int32 references; unchanged masks and labels excluded from both sides; no physical traffic measurement",
                source_sha256={str(p.resolve()):sha256(p) for p in files},
                seconds=time.perf_counter()-start)
    (args.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="complete",dataset=args.dataset,seconds=result["seconds"])),flush=True)


if __name__ == "__main__":
    main()
