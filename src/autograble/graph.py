from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .types import Stage1Result


def build_hetero_graph(df: pd.DataFrame, result: Stage1Result) -> HeteroData:
    """
    Build a bipartite HeteroData graph from the columns selected by Stage 1.

    Node types:
      "row"  — one node per row in df
      <col>  — one node type per selected column; one node per unique value in that column

    Edge types (one pair per selected column):
      ("row", "has", <col>)     — row i → value-node j when df[col][i] == value_j
      (<col>, "rev_has", "row") — reverse

    Value-node identifier:
      data[col].values — list of the original values (str representations), in node-index order
      NaN is represented as the string "__NaN__"
    """
    data = HeteroData()
    data["row"].num_nodes = len(df)

    for col in result.selected_cols:
        series = df[col]

        # Build value → node-index mapping; NaN gets its own node
        unique_vals = series.unique()  # includes NaN if present
        val_to_idx: dict = {
            (None if pd.isna(v) else v): i for i, v in enumerate(unique_vals)
        }

        data[col].num_nodes = len(unique_vals)
        data[col].values = [
            "__NaN__" if pd.isna(v) else str(v) for v in unique_vals
        ]

        # Build edge index (row-node → value-node)
        src, dst = [], []
        for row_i, v in enumerate(series):
            key = None if pd.isna(v) else v
            if key in val_to_idx:
                src.append(row_i)
                dst.append(val_to_idx[key])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["row", "has", col].edge_index = edge_index
        data[col, "rev_has", "row"].edge_index = edge_index.flip(0)

    return data
