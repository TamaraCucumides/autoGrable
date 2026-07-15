from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    target_column: Optional[str] = None,
    task: str = "classification",
) -> Tuple[HeteroData, Dict[str, List[str]]]:
    """
    Build a bipartite HeteroData graph over a set of selected columns.

    Node types:
      "row"  — one node per row in df
      <col>  — one node type per selected column; one node per unique value in that column

    Edge types (one pair per selected column):
      ("row", "has", <col>)     — row i → value-node j when df[col][i] == value_j
      (<col>, "rev_has", "row") — reverse

    Value-node identifier:
      data[col].x — [num_nodes, 1] float32 tensor, all zeros. Value nodes are
      categorical and have no tabular features of their own; message passing
      (not x) is what differentiates them. Kept 2D float32 rather than
      omitted so every node type has a uniformly-shaped, sampler-safe x.
      The human-readable values themselves (str representations, NaN as
      "__NaN__") are returned separately as value_vocab[col], in node-index
      order — kept off the HeteroData node stores because PyG's sampling
      loaders (e.g. NeighborLoader) index-select every node-store attribute
      by the sampled node ids, which requires a tensor and breaks (or
      silently misbehaves) on a plain list of strings.

    Row-node features (data["row"].x), when produced, come from tabularising
    other_columns via make_tabular_features (numeric / one-hot); already a
    2D float32 tensor of shape [n_rows, D].

    Row-node target (data["row"].y), when target_column is given, is read
    directly from df[target_column] — it must already be numerically encoded
    (e.g. via sklearn's LabelEncoder, fit once on the training split and
    reused across val/test) since each split is built into its own
    independent graph and encoding a target per-split here could assign
    different ids to the same class across splits. dtype follows task:
    torch.long for "classification" (CrossEntropyLoss), torch.float32 for
    "regression" or binary classification with BCEWithLogitsLoss.

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
        target_column:   Optional column stored as data["row"].y. Must already
                          be numerically encoded (see note above). When None,
                          no y is added.
        task:            "classification" (data["row"].y -> torch.long, for
                          CrossEntropyLoss) or "regression" (data["row"].y ->
                          torch.float32); shape [n_rows] either way, matching
                          fit_refinement's loss_fn(pred, y) usage. Ignored
                          when target_column is None.

    Returns:
        (data, value_vocab) where value_vocab maps each selected column to
        its list of value strings in node-index order (value_vocab[col][i]
        is the human-readable identity of value node i of that column).
    """
    data = HeteroData()
    data["row"].num_nodes = len(df)
    value_vocab: Dict[str, List[str]] = {}

    if other_columns:
        data["row"].x = make_tabular_features(
            df[list(other_columns)],
            numeric_fill=numeric_fill,
            encode_categoricals=encode_categoricals,
            max_cardinality=max_cardinality,
        )  # [n_rows, D]

    if temporal_column is not None:
        data["row"].time = _encode_temporal(df, temporal_column)  # [n_rows]

    if target_column is not None:
        y_dtype = torch.long if task == "classification" else torch.float32
        data["row"].y = torch.tensor(df[target_column].values, dtype=y_dtype)  # [n_rows]

    for col in selected_cols:
        series = df[col]

        # Build value → node-index mapping; NaN gets its own node
        unique_vals = series.unique()  # includes NaN if present
        val_to_idx: dict = {
            (None if pd.isna(v) else v): i for i, v in enumerate(unique_vals)
        }

        data[col].num_nodes = len(unique_vals)
        value_vocab[col] = [
            "__NaN__" if pd.isna(v) else str(v) for v in unique_vals
        ]
        # Value nodes are categorical and have no tabular features of their
        # own; kept as a zero-filled 2D float32 tensor (rather than omitted)
        # so every node type has a uniformly-shaped, sampler-safe x.
        data[col].x = torch.zeros((len(unique_vals), 1), dtype=torch.float32)

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

    return data, value_vocab
