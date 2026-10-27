"""Freeze a small same-tree input window; no device access or job submission."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

from benchmark_scripts.research.feature_sharing.device.core import SharedTreeModel, file_hash
from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["arxiv", "products"], default="arxiv")
    parser.add_argument("--modelzoo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fanouts", type=int, nargs="+", default=[15, 10, 5])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--capacity", type=int, help="Fixed rows including zero; default covers this frozen window")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.hidden, *args.fanouts) < 1 or not 0 <= args.dropout < 1:
        parser.error("positive sizes and dropout in [0,1) required")
    sys.path.insert(0, str(args.modelzoo_root/"src"))
    from benchmark_scripts.research.logs.local_factorial_sampler import FixedCyclicTreeProvider
    from benchmark_scripts.research.logs.local_ogb_graph import load_local_ogb_graph
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import GraphSAGENeighborSamplerDataset
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    graph = load_local_ogb_graph("ogbn-"+args.dataset, args.modelzoo_root/
                                "src/cerebras/modelzoo/models/gnn/data/datasets")
    source = graph.pop("metadata")
    source["dataset"] = args.dataset
    source["num_classes"] = 40 if args.dataset == "arxiv" else 47
    dataset = GraphSAGENeighborSamplerDataset(**graph, fanouts=args.fanouts,
        batch_size=args.batch_size, shuffle=True, pad_id=0, seed=args.seed)
    provider = FixedCyclicTreeProvider(dataset, seed=args.seed)
    indexes = sorted({0, min(1, len(dataset)-1), len(dataset)-1})
    args.output.mkdir(parents=True, exist_ok=False)
    rows, files, maximum = [], {}, 0
    for serial, index in enumerate(indexes):
        roots = np.zeros(args.batch_size, dtype=np.int64)
        active = np.zeros(args.batch_size, dtype=bool)
        real = dataset._ordered_targets[index*args.batch_size:(index+1)*args.batch_size]
        roots[:len(real)], active[:len(real)] = real, True
        nodes, masks, neighbors = (dataset._sample_layers(roots, active) if provider is None
                                   else provider.tree(index))
        unique = np.unique(np.concatenate([a[m] for a,m in zip(nodes,masks)]))
        table = torch.zeros(len(unique)+1, dataset.features.shape[-1], dtype=torch.float16)
        table[1:] = dataset.features[torch.from_numpy(unique)]
        refs = []
        for a,m in zip(nodes,masks):
            ref = np.zeros(a.size, dtype=np.int32)
            ref[m] = np.searchsorted(unique,a[m]).astype(np.int32)+1
            # Exact native feature lookup with the original padding contract.
            torch.testing.assert_close(table[torch.from_numpy(ref).long()],
                dataset.features[torch.from_numpy(a)] * torch.from_numpy(m).reshape(-1,1), rtol=0, atol=0)
            refs.append(torch.from_numpy(ref))
        labels = dataset.labels[torch.from_numpy(roots)].clone()
        labels[~torch.from_numpy(active)] = 0
        sample=dict(table=table, refs=tuple(refs), labels=labels, root_ids=torch.from_numpy(roots),
            node_masks=tuple(torch.from_numpy(m).reshape(args.batch_size,-1) for m in masks),
            neighbor_masks=tuple(torch.from_numpy(m).reshape(args.batch_size,-1,m.shape[-1]) for m in neighbors),
            target_mask=torch.from_numpy(active), sample_id=torch.tensor([serial],dtype=torch.int32))
        name=f"batch-{serial:03d}.pt"
        torch.save(sample,args.output/name)
        files[name]=file_hash(args.output/name)
        maximum=max(maximum,len(unique)+1)
        rows.append(dict(file=name, source_batch_index=index, valid_targets=len(real), unique_nodes=len(unique)))
    capacity=args.capacity or maximum
    if capacity < maximum:
        raise ValueError(f"capacity {capacity} < required {maximum}; no truncated input written as ready")
    config=dict(input_dim=dataset.features.shape[1], hidden_dim=args.hidden,
                num_layers=len(args.fanouts), dropout=args.dropout, aggregator="mean", num_classes=source["num_classes"])
    torch.manual_seed(args.seed)
    model=SharedTreeModel(config,args.batch_size,"dense")
    torch.save(model.state_dict(),args.output/"initial.pt")
    files["initial.pt"]=file_hash(args.output/"initial.pt")
    native=Path(inspect.getfile(GraphSAGE)).resolve().parents[1]
    metadata=dict(version=1, status="prepared", source=source, logical_batch_size=args.batch_size,
                  sdk_batch_size=1, fanouts=args.fanouts, capacity=capacity, seed=args.seed,
                  model_config=config,batches=rows,files=files,
                  native_sha256={name:file_hash(native/name) for name in ("architectures/graphsage.py", "data_processing/batches.py", "task/loss.py")},
                  scope="frozen first two and tail batches; repeated window is not a training epoch; no online sampling")
    (args.output/"input.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)


if __name__=="__main__":
    main()
