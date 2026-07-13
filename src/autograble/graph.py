from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .preprocess import make_tabular_features


def _encode_temporal(df: pd.DataFrame, temporal_column: str) -> torch.Tensor:
    """
    Convert a datetime column into raw seconds-since-epoch, one value per row.
    NaT becomes NaN — left for the consumer (loader/model) to handle, not
    imputed here, since this is metadata rather than a model-ready feature.
    """
    t = pd.to_datetime(df[temporal_column])
    t_seconds = t.view("int64").astype(float) / 1e9  # ns -> s since epoch
    t_seconds[t.isna()] = float("nan")

    return torch.tensor(t_seconds.values, dtype=torch.float)  # [n_rows]


def build_hetero_graph(
    df: pd.DataFrame,
    selected_cols: List[str],
    other_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    numeric_fill: str = "mean",
    encode_categoricals: bool = True,
    max_cardinality: int = 50,
) -> HeteroData:
    """
    Build a bipartite HeteroData graph over a set of selected columns.

    Node types:
      "row"  — one node per row in df
      <col>  — one node type per selected column; one node per unique value in that column

    Edge types (one pair per selected column):
      ("row", "has", <col>)     — row i → value-node j when df[col][i] == value_j
      (<col>, "rev_has", "row") — reverse

    Value-node identifier:
      data[col].values — list of the original values (str representations), in node-index order
      NaN is represented as the string "__NaN__"

    Row-node features (data["row"].x), when produced, come from tabularising
    other_columns via make_tabular_features (numeric / one-hot).

    Row-node temporal metadata (data["row"].time), when produced, is a
    [n_rows] tensor of seconds-since-epoch — kept separate from data["row"].x
    so a downstream loader/model can use it explicitly (e.g. to order or mask
    rows by time) rather than it being an opaque column buried in the tabular
    feature block.

    Args:
        df:              DataFrame for this split.
        selected_cols:   Columns to encode as value-node types + row<->value edges.
        other_columns:   Optional columns to tabularise into row-node features
                          (data["row"].x). When None, no such features are added.
        temporal_column:  Optional datetime column stored as row-node metadata
                          (data["row"].time), for the consuming loader/model to
                          use however it needs to (e.g. to prevent leakage). Not
                          folded into data["row"].x and not imputed/normalised
                          here.
        numeric_fill:        Passed through to make_tabular_features.
        encode_categoricals:  Passed through to make_tabular_features.
        max_cardinality:      Passed through to make_tabular_features.
    """
    data = HeteroData()
    data["row"].num_nodes = len(df)

    if other_columns:
        data["row"].x = make_tabular_features(
            df[list(other_columns)],
            numeric_fill=numeric_fill,
            encode_categoricals=encode_categoricals,
            max_cardinality=max_cardinality,
        )  # [n_rows, D]

    if temporal_column is not None:
        data["row"].time = _encode_temporal(df, temporal_column)  # [n_rows]

    for col in selected_cols:
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
