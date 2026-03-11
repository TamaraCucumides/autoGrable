"""
End-to-end autoGrable example
==============================

Stage 1  — greedy backward elimination to find structurally relevant columns
Stage 2  — bipartite Gated GNN that learns which of those columns actually matter
           for the prediction task, using raw tabular features in the head
"""

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # only needed to encode y

from autograble import (
    Stage1Config, fit_structure_stage1,
    build_hetero_graph,
    Stage2Config, fit_gated_gnn,
    make_tabular_features,
    gate_summary,
)

# ---------------------------------------------------------------------------
# 0. Load your data
# ---------------------------------------------------------------------------
# Replace this with your own DataFrame.
# df must contain a target column and candidate attribute columns.

df = pd.read_csv("your_data.csv")
TARGET = "target"

# ---------------------------------------------------------------------------
# 1. Stage 1 — structural column selection
# ---------------------------------------------------------------------------
# cardinality_encoding=True replaces each value with how many rows share it,
# so Stage 1 focuses on graph-structural signals rather than specific identities.

s1_config = Stage1Config(
    y_col=TARGET,
    cardinality_encoding=True,  # replace values with their peer-group size
    lambda_=1.0,                # regularisation: penalise complex partitions
    val_frac=0.2,
    random_state=42,
)

result = fit_structure_stage1(df, s1_config)

print("Selected columns:", result.selected_cols)
print("Dropped  columns:", result.dropped_cols)

# ---------------------------------------------------------------------------
# 2. Build the heterogeneous graph
# ---------------------------------------------------------------------------
# Uses the ORIGINAL df (not the cardinality-encoded one).
# Node types : "row"  — one node per row
#              <col>  — one node per unique value in that column
# Edge types : ("row", "has", <col>)     and its reverse

graph = build_hetero_graph(df, result)
print(graph)

# ---------------------------------------------------------------------------
# 3. Prepare inputs for Stage 2
# ---------------------------------------------------------------------------

# 3a. Target labels as a Long tensor (for classification)
le = LabelEncoder()
y = torch.tensor(le.fit_transform(df[TARGET]), dtype=torch.long)

# 3b. Raw tabular features for the prediction head.
#     Exclude the target and the structural columns that are already in the graph.
x_tab = make_tabular_features(
    df,
    exclude_cols=[TARGET] + result.selected_cols,
)
# x_tab shape: [num_rows, num_remaining_features]

# 3c. Optional train / val split masks
n = len(df)
perm = torch.randperm(n)
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
train_mask[perm[:int(0.8 * n)]] = True
val_mask[perm[int(0.8 * n):]]   = True

# ---------------------------------------------------------------------------
# 4. Stage 2 — Gated GNN
# ---------------------------------------------------------------------------
# The GNN propagates over the bipartite graph and learns a scalar gate per
# column edge type.  Gates close to 0 mean the model ignores that structure.
# The head combines the GNN embedding with x_tab, keeping the GNN focused
# purely on structural signals.

s2_config = Stage2Config(
    hidden_dim=64,
    num_layers=3,
    dropout=0.1,
    lr=1e-3,
    epochs=200,
    task="classification",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

s2 = fit_gated_gnn(
    graph,
    y,
    config=s2_config,
    x_tab=x_tab,
    train_mask=train_mask,
    val_mask=val_mask,
)

# ---------------------------------------------------------------------------
# 5. Inspect which structural columns the GNN found useful
# ---------------------------------------------------------------------------

summary = gate_summary(s2, threshold=0.2)
print("\nEdge gate summary (sorted, most ignored first):")
print(summary)
# Example output:
#      column    gate    status
# 0       Age  0.0312   ignored
# 1    Income  0.1841   ignored
# 2    Gender  0.3107      weak
# 3      City  0.8743    active

# ---------------------------------------------------------------------------
# 6. Get predictions
# ---------------------------------------------------------------------------

s2.model.eval()
with torch.no_grad():
    logits = s2.model(graph.to(s2_config.device), x_tab.to(s2_config.device))

preds = logits.argmax(dim=-1).cpu()
labels = le.inverse_transform(preds.numpy())
print("\nFirst 10 predictions:", labels[:10])
