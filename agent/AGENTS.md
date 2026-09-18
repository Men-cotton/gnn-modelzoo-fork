# Agent reference index

Keep this directory limited to current, reusable knowledge. Put regression behavior in the existing GNN tests; do not retain completed plans, work reports, transient logs or resolved-defect inventories here.

Read only the reference relevant to the task:

- [Pipeline contracts](reference/pipeline-contracts.md): loader ownership, masking, GPU configuration, supported extensions, and validation commands. Read when changing GNN runtime behavior or the SDK.
- [Masked-loss lowering fixture](reference/fixtures/masked-loss.mlir): a small equivalent forward/backward graph for local SDK conversion. Use with the pipeline reference; it is not a full model export.
- [Compiler artifact reference](reference/compiler-artifacts.md): archive locations, interpretation limits, unsupported sparse shapes and SDK loss counterexamples. Read when inspecting compiler output or reconsidering an SDK workaround.
- [HPC Asia revalidation](research/hpcasia-revalidation.md): the unresolved relationship between the manuscript and its recorded experiments. Read for paper claims or replacement measurements; update or remove it when that dependency is closed.

Operational instructions already live in the [GNN docs](../src/cerebras/modelzoo/models/gnn/docs/). Environment requirements come from [pyproject.toml](../pyproject.toml), not historical reports.

This compact index applies the [GPT-6 Astra guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra) on auditing conflicting instructions and keeping verification proportional to the change. It does not add mandatory planning templates, automatic commit instructions or duplicate general coding rules.
