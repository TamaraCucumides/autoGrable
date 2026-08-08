# AutoGrable

autoGrable induces a structural partition over a table by greedily selecting the columns
whose equality best explains the target, trading off validation loss against partition
complexity. It's a complete, standalone algorithm — the result is usable on its own.

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
)

# 1. autoGrable: select structurally relevant columns
result = fit_autograble(df, AutoGrableConfig(
    y_col="target",
    cardinality_encoding=True,  # replace values with peer-group size
))

# 2. Build bipartite heterogeneous graph (row nodes ↔ value nodes per column)
# other_columns become row-node features (data["row"].x); temporal_column
# (optional) is stored separately as row-node metadata (data["row"].time)
# for the training code to use explicitly, e.g. to prevent leakage.

other_columns = [c for c in df.columns if c not in ["target"] + result.selected_cols]
graph = build_hetero_graph(df, result.selected_cols, other_columns=other_columns,
                            temporal_column="date")

# 3. Prepare labels for the prediction head
y = torch.tensor(df["target"].values, dtype=torch.long)

# 4. Use your favourite Graph Learning model to predict :)

```
