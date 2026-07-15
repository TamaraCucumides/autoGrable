from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .preprocess import make_tabular_features


NAT_TIME = torch.iinfo(torch.long).min


def _encode_temporal(df: pd.DataFrame, temporal_column: str) -> torch.Tensor:
    """
    Convert a datetime column into whole seconds-since-epoch (int64/long),
    one value per row. int64 (rather than float) is required by PyG's
    temporal neighbor loaders. NaT becomes NAT_TIME — the minimum representable
    int64 — so a row with a missing timestamp always sorts before every real
    timestamp and never spuriously satisfies "neighbor time <= seed time"
    under temporal neighbor sampling; not imputed to a real time since this
    is metadata rather than a model-ready feature.
    """
    t = pd.to_datetime(df[temporal_column])
    t_seconds = t.view("int64") // 1_000_000_000  # ns -> s since epoch
    t_seconds[t.isna()] = NAT_TIME

    return torch.tensor(t_seconds.values, dtype=torch.long)  # [n_rows]


def build_hetero_graph(
    df: pd.DataFrame,
    selected_cols: List[str],
    other_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    zero_time_value_nodes: bool = False,
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
      data[col].x — [num_nodes] long tensor of node-id indices (0..num_nodes-1),
      for an nn.Embedding lookup in the hetero GNN, since value nodes are
      categorical and have no tabular features of their own.

    Row-node features (data["row"].x), when produced, come from tabularising
    other_columns via make_tabular_features (numeric / one-hot).

    Row-node temporal metadata (data["row"].time), when produced, is a
    [n_rows] long tensor of whole seconds-since-epoch — kept separate from
    data["row"].x so a downstream loader/model can use it explicitly (e.g.
    PyG's temporal NeighborLoader, which requires an int64 time_attr) rather
    than it being an opaque column buried in the tabular feature block.
    Missing timestamps (NaT) are encoded as NAT_TIME (torch.iinfo(int64).min)
    rather than NaN, since int64 can't represent NaN; this sentinel sorts
    before every real timestamp so it never spuriously satisfies "neighbor
    time <= seed time" under temporal sampling.

    Args:
        df:              DataFrame for this split.
        selected_cols:   Columns to encode as value-node types + row<->value edges.
        other_columns:   Optional columns to tabularise into row-node features
                          (data["row"].x). When None, no such features are added.
        temporal_column:  Optional datetime column stored as row-node metadata
                          (data["row"].time), for the consuming loader/model to
                          use however it needs to (e.g. to prevent leakage). Not
                          folded into data["row"].x and not imputed/normalised
                          here (NaT -> NAT_TIME sentinel; see above).
        zero_time_value_nodes: When True (and temporal_column is set), give
                          every non-row (value) node a time of 0 — i.e.
                          data[col].time = zeros (long). Since value nodes always
                          predate any row timestamp, this keeps them eligible
                          as neighbors under temporal neighbor sampling
                          (which typically requires neighbor time <= seed
                          time), regardless of when the row itself occurred.
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
        # Node-id index per value node, for an nn.Embedding lookup in the
        # hetero GNN (value nodes are categorical — no tabular features).
        data[col].x = torch.arange(len(unique_vals), dtype=torch.long)

        if zero_time_value_nodes and temporal_column is not None:
            data[col].time = torch.zeros(len(unique_vals), dtype=torch.long)

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
