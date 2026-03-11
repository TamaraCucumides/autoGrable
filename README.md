# autoGrable

Stage-1 structure selection for table-to-graph learning.

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
    Stage1Config, fit_structure_stage1,
    build_hetero_graph,
    Stage2Config, fit_gated_gnn,
    make_tabular_features,
    gate_summary,
)

# 1. Select structurally relevant columns
result = fit_structure_stage1(df, Stage1Config(
    y_col="target",
    cardinality_encoding=True,  # replace values with peer-group size
))

# 2. Build bipartite heterogeneous graph (row nodes ↔ value nodes per column)
graph = build_hetero_graph(df, result)

# 3. Prepare labels and tabular features for the prediction head
y     = torch.tensor(df["target"].values, dtype=torch.long)
x_tab = make_tabular_features(df, exclude_cols=["target"] + result.selected_cols)

# 4. Train the Gated GNN — learns a scalar gate per column edge type
s2 = fit_gated_gnn(graph, y, config=Stage2Config(), x_tab=x_tab,
                   train_mask=train_mask, val_mask=val_mask)

# 5. See which structural columns the model found relevant
print(gate_summary(s2))
#      column    gate    status
# 0       Age  0.0312   ignored
# 1    Income  0.1841   ignored
# 2    Gender  0.3107      weak
# 3      City  0.8743    active

# 6. Predict
s2.model.eval()
logits = s2.model(graph, x_tab)
preds  = logits.argmax(dim=-1)
```