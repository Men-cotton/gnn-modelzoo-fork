"""CPU-only probes of optimizer, model and sampler differences; no jobs."""
import numpy as np
import torch
import cerebras.pytorch as cs
from cerebras.modelzoo.common.optim_utils import configure_param_groups
from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import GraphSAGENeighborSamplerDataset
from cerebras.modelzoo.models.gnn.reference.pyg.model import get_model
from cerebras.modelzoo.models.gnn.reference.pyg.data import make_loaders
from cerebras.modelzoo.models.gnn.tools.learning_campaign import shared_base
from torch_geometric.data import Data

torch.set_num_threads(1)
cs.backend('CPU')
for grad in (1e-6, 1e-3, 1.0):
    p, q = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(2)]
    a = cs.optim.AdamW([p], lr=.003, eps=1e-6, weight_decay=0)
    b = torch.optim.AdamW([q], lr=.003, eps=1e-6, weight_decay=0, foreach=False)
    p.grad = torch.tensor([grad])
    q.grad = torch.tensor([grad])
    a.step()
    b.step()
    print('OPTIMIZER_ONE_STEP', {'gradient':grad, 'cs_weight':p.item(), 'torch_weight':q.item(), 'cs_update':1-p.item(), 'torch_update':1-q.item()})
    assert p.item() != q.item()

for dataset in ('arxiv', 'products'):
    cfg = shared_base(dataset)
    arch = cfg['trainer']['init']['model']['architecture']
    fixed = GraphSAGE(arch['n_feat'], arch['hidden_dim'], 3, .5, 'mean', arch['n_class'])
    pyg = get_model(cfg)
    def n(m):
        return sum(p.numel() for p in m.parameters())
    print('PARAMETERS',dataset, {'cs':n(fixed), 'pyg':n(pyg), 'difference':n(fixed)-n(pyg), 'cs_biases':[name for name,p in fixed.named_parameters() if 'bias' in name], 'pyg_biases':[name for name,p in pyg.named_parameters() if 'bias' in name]})
    assert n(fixed)-n(pyg)==3072
    groups = configure_param_groups(fixed, {'weight_decay':.0005})
    by_id = {id(p):g['weight_decay'] for g in groups for p in g['params']}
    print('DECAY', dataset, {name:by_id[id(p)] for name,p in fixed.named_parameters()})
    assert all(by_id[id(p)]==0 for name,p in fixed.named_parameters() if 'bias' in name)
    b = torch.optim.AdamW(pyg.parameters(), lr=.003, eps=1e-6, weight_decay=.0005)
    assert all(g['weight_decay']==.0005 for g in b.param_groups)

# A complete small graph makes changing neighbor subsets easy to observe.
n=12
edges=torch.tensor([(i,j) for i in range(n) for j in range(n) if i!=j]).T.contiguous()
features=torch.arange(n,dtype=torch.float32).view(-1,1)
labels=torch.zeros(n,dtype=torch.long)
fixed = GraphSAGENeighborSamplerDataset(features,edges,labels,torch.ones(n,dtype=torch.bool),[3],4,shuffle=True,pad_id=0,seed=42,edge_index_is_undirected=True)
same = all(torch.equal(a,b) for a,b in zip(fixed[0]['node_features'],fixed[0]['node_features']))
print('FIXED_REPEAT_IDENTICAL',same)
assert same
# Vary seed and enumerate first-hop selections for a node of degree six.
selected=[tuple(GraphSAGENeighborSamplerDataset(features,edges,labels,torch.ones(n,dtype=torch.bool),[3],4,shuffle=False,pad_id=0,seed=s,edge_index_is_undirected=True)._deterministic_choice(np.arange(6),None,3,0,0)) for s in range(12)]
print('FIXED_CHOICES_6_NEIGHBORS_3_FANOUT',sorted(set(selected)))
assert len(set(selected)) == 6  # 6 cyclic windows versus C(6,3)=20 subsets.
cfg = shared_base('arxiv')
loader_cfg=cfg['trainer']['fit']['train_dataloader']
loader_cfg.update(num_workers=0,persistent_workers=False,batch_size=4,fanouts=[3])
cfg['trainer']['fit']['val_dataloader']=None
train,_=make_loaders(Data(x=features,edge_index=edges,y=labels),{'train':torch.arange(n)},cfg)
epochs = [[batch.n_id.tolist() for batch in train] for _ in range(2)]
print('PYG_EPOCHS',epochs)
print('PYG_EPOCHS_DIFFER',epochs[0]!=epochs[1])
assert epochs[0]!=epochs[1]
print('PROBES_PASSED')
