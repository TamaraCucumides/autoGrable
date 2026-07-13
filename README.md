# autoGrable

autoGrable induces a structural partition over a table by greedily selecting the columns
whose equality best explains the target, trading off validation loss against partition
complexity. It's a complete, standalone algorithm — the result is usable on its own.

An optional refinement stage builds a heterogeneous graph from the selected structure
and trains a parametric Gated GNN on top of it, for cases where the discrete partition
alone isn't expressive enough.

## Installation

### From GitHub

**CPU only** — torch installs automatically:

```bash
pip install git+https://github.com/your-username/autoGrable.git
```

**GPU (CUDA)** — install torch with the right CUDA version first, then install autoGrable:

```bash
# example for CUDA 12.1 — see https://pytorch.org/get-started/locally for other versions
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/your-username/autoGrable.git
```

### Local development

```bash
git clone https://github.com/your-username/autoGrable.git
cd autoGrable
pip install torch --index-url https://download.pytorch.org/whl/cu121  # skip for CPU
pip install -e ".[dev]"
```

Or use the provided `requirements.txt` for a pinned CPU environment:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

A complete runnable example is in [examples/usage.py](examples/usage.py).

### Quick overview

```python
from autograble import (
    AutoGrableConfig, fit_autograble,
    build_hetero_graph,
    RefinementConfig, fit_gated_gnn,
    gate_summary,
)

# 1. autoGrable: select structurally relevant columns
result = fit_autograble(df, AutoGrableConfig(
    y_col="target",
    cardinality_encoding=True,  # replace values with peer-group size
))

# 2. (optional) Build bipartite heterogeneous graph (row nodes ↔ value nodes per column)
# other_columns become row-node features (data["row"].x); temporal_column
# (optional) is stored separately as row-node metadata (data["row"].time)
# for the training code to use explicitly, e.g. to prevent leakage.
other_columns = [c for c in df.columns if c not in ["target"] + result.selected_cols]
graph = build_hetero_graph(df, result.selected_cols, other_columns=other_columns,
                            temporal_column="date")

# 3. Prepare labels for the prediction head
y = torch.tensor(df["target"].values, dtype=torch.long)

# 4. (optional) Refine: train the Gated GNN — learns a scalar gate per column edge type
refinement = fit_gated_gnn(graph, y, config=RefinementConfig(),
                            graph_val=graph_val, y_val=y_val)

# 5. See which structural columns the model found relevant
print(gate_summary(refinement))
#      column    gate    status
# 0       Age  0.0312   ignored
# 1    Income  0.1841   ignored
# 2    Gender  0.3107      weak
# 3      City  0.8743    active

# 6. Predict
refinement.model.eval()
logits = refinement.model(graph)
preds  = logits.argmax(dim=-1)
```