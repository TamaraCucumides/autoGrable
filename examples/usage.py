"""
End-to-end autoGrable example (inductive, 3-split)
====================================================

Stage 1  — greedy backward elimination on df_train; df_val used for lambda selection
Stage 2  — inductive Gated GNN trained on graph_train, evaluated on graph_val
           each split is its own independent graph (no shared node IDs)

Row nodes carry tabular features (non-selected columns) as node features.
Value nodes carry no features — the GNN learns per-column prototype vectors.
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # only needed to encode y

from autograble import (
    Stage1Config, fit_structure_stage1,
    build_hetero_graph,
    Stage2Config, fit_stage2,
    make_tabular_features,
    gate_summary,
    MODELS,
)
from autograble.models import HeteroGatedGNN

# ---------------------------------------------------------------------------
# 0. Load and split your data
# ---------------------------------------------------------------------------

df = pd.read_csv("your_data.csv")
TARGET = "target"

# Three independent splits — stratify on target if classification
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test   = train_test_split(df_temp, test_size=0.5, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1. Stage 1 — structural column selection
# ---------------------------------------------------------------------------
# df_val is passed explicitly so Stage 1 skips its internal split and uses
# the same validation set as Stage 2.
# Cardinality maps are computed on df_train and applied to df_val automatically.

s1_config = Stage1Config(
    y_col=TARGET,
    cardinality_encoding=True,  # replace values with peer-group size
    lambda_=1.0,
    random_state=42,
)

result = fit_structure_stage1(df_train, s1_config, df_val=df_val)

print("Selected columns:", result.selected_cols)
print("Dropped  columns:", result.dropped_cols)

# ---------------------------------------------------------------------------
# 2. Build tabular features for row nodes (non-selected, non-target columns)
# ---------------------------------------------------------------------------
# These are embedded directly as row node features in each graph.
# The GNN uses them for row initialisation; value nodes use learnable prototypes.

exclude = [TARGET] + result.selected_cols

x_train = make_tabular_features(df_train, exclude_cols=exclude)
x_val   = make_tabular_features(df_val,   exclude_cols=exclude)
x_test  = make_tabular_features(df_test,  exclude_cols=exclude)

# ---------------------------------------------------------------------------
# 3. Build one graph per split (inductive)
# ---------------------------------------------------------------------------
# Each graph is independent — val/test may contain values never seen in train.
# x_tab is stored as data["row"].x for row-node initialisation in the GNN.

graph_train = build_hetero_graph(df_train, result, x_tab=x_train)
graph_val   = build_hetero_graph(df_val,   result, x_tab=x_val)
graph_test  = build_hetero_graph(df_test,  result, x_tab=x_test)

print(graph_train)

# ---------------------------------------------------------------------------
# 4. Prepare labels
# ---------------------------------------------------------------------------

le = LabelEncoder().fit(df_train[TARGET])

def encode_y(df):
    return torch.tensor(le.transform(df[TARGET]), dtype=torch.long)

y_train = encode_y(df_train)
y_val   = encode_y(df_val)
y_test  = encode_y(df_test)

# ---------------------------------------------------------------------------
# 5. Stage 2 — inductive Gated GNN
# ---------------------------------------------------------------------------

s2_config = Stage2Config(
    hidden_dim=64,
    num_layers=3,
    dropout=0.1,
    lr=1e-3,
    epochs=200,
    task="classification",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

s2 = fit_stage2(
    graph_train, y_train,
    config=s2_config,
    model_cls=HeteroGatedGNN,       # swap for any model in MODELS
    graph_val=graph_val,
    y_val=y_val,
)

# ---------------------------------------------------------------------------
# 6. Inspect learned structure relevance
# ---------------------------------------------------------------------------

print("\nEdge gate summary (sorted, most ignored first):")
print(gate_summary(s2, threshold=0.2))

# ---------------------------------------------------------------------------
# 7. Evaluate on held-out test split
# ---------------------------------------------------------------------------

dev = torch.device(s2_config.device)
s2.model.eval()
with torch.no_grad():
    logits = s2.model(graph_test.to(dev))

preds  = logits.argmax(dim=-1).cpu()
labels = le.inverse_transform(preds.numpy())
print("\nFirst 10 test predictions:", labels[:10])
