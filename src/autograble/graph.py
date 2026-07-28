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


def build_hetero_graph_from_joined_table(
    df: pd.DataFrame,
    primary_key: str,
    selected_cols: List[str],
    temporal_columns: Optional[Dict[str, str]] = None,
    temporal_column: Optional[str] = None,
    other_columns: Optional[List[str]] = None,
    numeric_fill: str = "mean",
    encode_categoricals: bool = True,
    max_cardinality: int = 50,
    target_column: Optional[str] = None,
    task: str = "classification",
) -> Tuple[HeteroData, Dict[str, List[str]]]:
    """
    Build a bipartite HeteroData graph for a "joined_table": a df where several
    rows can share the same primary_key (e.g. an entity table joined against a
    fact/event table, so one customer id appears once per transaction row).

    Unlike build_hetero_graph (one node per row), here:
      "row"  — one node per UNIQUE value of primary_key. Rows sharing a
               primary_key are the same node, not duplicates.
      <col>  — one node type per selected column, one node per unique value
               of that column across ALL rows of df (not deduped by
               primary_key — repeated rows are expected to carry different
               values).

    Edge types (one pair per selected column):
      ("row", "has", <col>)     — one edge per df ROW (not per unique
                                  primary_key): row-node for that
                                  primary_key -> value-node for df[col] at
                                  that row. A primary_key repeated N times
                                  yields (up to) N parallel edges into <col>,
                                  one per occurrence — this is how "id X had
                                  value Y at this time" is represented for
                                  entities with several events.
      (<col>, "rev_has", "row") — reverse, same per-edge time.

    Edge-level time (data["row", "has", col].time / the reverse edge's
    .time), a [n_edges] int64 tensor of whole seconds-since-epoch aligned to
    edge order: since each selected column can be tied to its own point in
    time (not one timestamp per row), time is attached to the edge — "id X
    has value Y at this time" — rather than to either endpoint node. Neither
    the "row" nodes nor the value nodes carry a .time here (contrast
    build_hetero_graph, where row.time is a node attribute); an entity's
    events are only orderable through its edges. Resolved per column via
    temporal_columns[col] if present, else the shared temporal_column
    fallback; if neither is given for a column, that edge type simply has no
    .time (untimed edges). NaT -> NAT_TIME sentinel, same convention as
    build_hetero_graph.

    Row-node features/target (data["row"].x / .y): built from other_columns /
    target_column taken from the FIRST occurrence of each primary_key (df,
    with duplicates on primary_key dropped, keep="first"). This assumes those
    columns are entity-level attributes constant across a given primary_key's
    repeated rows (as they would be after a join) — if they vary across
    occurrences, only the first row's value is used.

    Args:
        df:              DataFrame for this split; may repeat primary_key.
        primary_key:     Column identifying the entity; "row" nodes are
                          deduped on this column (first-appearance order).
        selected_cols:   Columns to encode as value-node types + per-row
                          timestamped edges.
        temporal_columns: Optional {col: timestamp_column} giving each
                          selected column its own time source. Columns not
                          present here fall back to temporal_column.
        temporal_column: Optional default timestamp column used for any
                          selected column not overridden in temporal_columns.
                          When a column resolves to no timestamp column at
                          all, its edges are left untimed.
        other_columns:   Optional columns tabularised (via
                          make_tabular_features, on the first-occurrence
                          frame) into data["row"].x. When None, no such
                          features are added.
        numeric_fill:        Passed through to make_tabular_features.
        encode_categoricals:  Passed through to make_tabular_features.
        max_cardinality:      Passed through to make_tabular_features.
        target_column:   Optional column (read from the first-occurrence
                          frame) stored as data["row"].y. Must already be
                          numerically encoded — see build_hetero_graph's note.
        task:            "classification" (torch.long) or "regression"
                          (torch.float32); see build_hetero_graph.

    Returns:
        (data, value_vocab), same shape/meaning as build_hetero_graph's
        return value.
    """
    data = HeteroData()
    value_vocab: Dict[str, List[str]] = {}
    temporal_columns = temporal_columns or {}

    pk_series = df[primary_key]
    unique_pks = pk_series.unique()  # first-appearance order
    pk_to_idx = {v: i for i, v in enumerate(unique_pks)}
    data["row"].num_nodes = len(unique_pks)

    # Entity-level frame: first occurrence of each primary_key, in node-index
    # order, used for row-node features/target (assumed constant per entity).
    entity_df = df.drop_duplicates(subset=primary_key, keep="first").set_index(
        primary_key, drop=False
    ).loc[unique_pks]

    if other_columns:
        data["row"].x = make_tabular_features(
            entity_df[list(other_columns)],
            numeric_fill=numeric_fill,
            encode_categoricals=encode_categoricals,
            max_cardinality=max_cardinality,
        )  # [n_entities, D]

    if target_column is not None:
        y_dtype = torch.long if task == "classification" else torch.float32
        data["row"].y = torch.tensor(entity_df[target_column].values, dtype=y_dtype)  # [n_entities]

    row_node_idx = pk_series.map(pk_to_idx).to_numpy()  # [n_rows], entity node per df row

    for col in selected_cols:
        series = df[col]

        unique_vals = series.unique()  # includes NaN if present
        val_to_idx: dict = {
            (None if pd.isna(v) else v): i for i, v in enumerate(unique_vals)
        }

        data[col].num_nodes = len(unique_vals)
        value_vocab[col] = [
            "__NaN__" if pd.isna(v) else str(v) for v in unique_vals
        ]
        data[col].x = torch.zeros((len(unique_vals), 1), dtype=torch.float32)

        # Build edge index (row-node -> value-node), one edge per df row.
        src, dst = [], []
        for row_i, v in enumerate(series):
            key = None if pd.isna(v) else v
            src.append(row_node_idx[row_i])
            dst.append(val_to_idx[key])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["row", "has", col].edge_index = edge_index
        data[col, "rev_has", "row"].edge_index = edge_index.flip(0)

        time_col = temporal_columns.get(col, temporal_column)
        if time_col is not None:
            edge_time = _encode_temporal(df, time_col)  # [n_rows], aligned to edge order
            data["row", "has", col].time = edge_time
            data[col, "rev_has", "row"].time = edge_time

    return data, value_vocab
