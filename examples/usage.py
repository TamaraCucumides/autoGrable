"""
End-to-end autoGrable example (inductive, 3-split)
====================================================

autoGrable  — greedy backward elimination on df_train; df_val used for lambda selection.
              This is the core algorithm: it induces a structural partition over the
              data and is usable on its own.
Refinement  — optional inductive Gated GNN trained on graph_train, evaluated on
              graph_val, each split is its own independent graph (no shared node IDs).
              Learns a parametric model on top of the structure autoGrable selected.

Row nodes carry tabular features (non-selected columns) as node features.
Value nodes carry no features — the GNN learns per-column prototype vectors.
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # only needed to encode y

from autograble import (
    AutoGrableConfig, fit_autograble,
    build_hetero_graph,
    RefinementConfig, fit_refinement,
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
# 1. autoGrable — structural column selection
# ---------------------------------------------------------------------------
# df_val is passed explicitly so autoGrable skips its internal split and uses
# the same validation set as the refinement stage.
# Cardinality maps are computed on df_train and applied to df_val automatically.

config = AutoGrableConfig(
    y_col=TARGET,
    cardinality_encoding=True,  # replace values with peer-group size
    lambda_=1.0,
    random_state=42,
)

result = fit_autograble(df_train, config, df_val=df_val)

print("Selected columns:", result.selected_cols)
print("Dropped  columns:", result.dropped_cols)

# ---------------------------------------------------------------------------
# 2. Prepare labels
# ---------------------------------------------------------------------------
# Fit the encoder once on df_train and reuse it for val/test: build_hetero_graph
# builds each split into its own independent graph, so encoding the target
# per-split would risk assigning different ids to the same class across splits.

le = LabelEncoder().fit(df_train[TARGET])
ENCODED_TARGET = f"{TARGET}__encoded"

for split in (df_train, df_val, df_test):
    split[ENCODED_TARGET] = le.transform(split[TARGET])

# ---------------------------------------------------------------------------
# 3. Build one graph per split (inductive)
# ---------------------------------------------------------------------------
# Each graph is independent — val/test may contain values never seen in train.
# other_columns are tabularised internally and stored as data["row"].x, used
# for row-node initialisation in the GNN. temporal_column (e.g. a transaction
# date) is stored separately as data["row"].time metadata — not folded into
# x — so training code can use it explicitly (e.g. to prevent leakage).
# target_column stores the pre-encoded label as data["row"].y.

other_columns = [
    c for c in df.columns if c not in [TARGET, ENCODED_TARGET] + result.selected_cols
]

graph_train, vocab_train = build_hetero_graph(
    df_train, result.selected_cols, other_columns=other_columns,
    target_column=ENCODED_TARGET, task="classification",
)
graph_val, vocab_val = build_hetero_graph(
    df_val, result.selected_cols, other_columns=other_columns,
    target_column=ENCODED_TARGET, task="classification",
)
graph_test, vocab_test = build_hetero_graph(
    df_test, result.selected_cols, other_columns=other_columns,
    target_column=ENCODED_TARGET, task="classification",
)

print(graph_train)

y_train = graph_train["row"].y
y_val   = graph_val["row"].y
y_test  = graph_test["row"].y

# ---------------------------------------------------------------------------
# 4. Refinement (optional) — inductive Gated GNN
# ---------------------------------------------------------------------------

refine_config = RefinementConfig(
    hidden_dim=64,
    num_layers=3,
    dropout=0.1,
    lr=1e-3,
    epochs=200,
    task="classification",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

refinement = fit_refinement(
    graph_train, y_train,
    config=refine_config,
    model_cls=HeteroGatedGNN,       # swap for any model in MODELS
    graph_val=graph_val,
    y_val=y_val,
)

# ---------------------------------------------------------------------------
# 5. Inspect learned structure relevance
# ---------------------------------------------------------------------------

print("\nEdge gate summary (sorted, most ignored first):")
print(gate_summary(refinement, threshold=0.2))

# ---------------------------------------------------------------------------
# 6. Evaluate on held-out test split
# ---------------------------------------------------------------------------

dev = torch.device(refine_config.device)
refinement.model.eval()
with torch.no_grad():
    logits = refinement.model(graph_test.to(dev))

preds  = logits.argmax(dim=-1).cpu()
labels = le.inverse_transform(preds.numpy())
print("\nFirst 10 test predictions:", labels[:10])
