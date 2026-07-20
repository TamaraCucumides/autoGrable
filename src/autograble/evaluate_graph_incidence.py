"""
J via the equality-projection partition.

For a value-exposed, column-restricted incidence grable, 1-WL on the row nodes
induces exactly the partition
        r ~ r'  <=>  r|S = r'|S          (S = selected_cols)
i.e. two rows share a block iff they agree on all selected columns. So we do NOT
build a graph or run colour refinement — we just group rows by their projection
onto S and score that partition with the same J as the general path.

    J(pi_S) = Risk_val(h_pi) + lambda * Omega(train, pi_S)

Why the general path's knobs disappear here:
  * n_rounds : irrelevant — by Lemma 4.1 the equality partition IS the WL fixpoint,
               independent of depth.
  * init_color_fn (feature vs structure) : irrelevant — value nodes are exposed as
               identities by construction, so the row partition is fixed by S alone.
  These collapse ONLY under the lemma's precondition: value nodes carry identities
  (c, a). If you bin values or don't expose value identities, this equivalence
  breaks and you must fall back to j_metric.compute_J. Binning is supported below
  via bin_fns (Def. 3.3), which just coarsens the projection.

Single-table only.
"""

from __future__ import annotations

from typing import Callable, Hashable, Mapping, Optional, Sequence

import pandas as pd

# reuse the shared, fully-specified pieces
from .evaluate_graph import (
    JResult,
    Label,
    Color,
    _stable_hash,
    fit_block_predictor,
    val_risk,
    occupancy,
)
from .utils import (
    train_val_split,
    safe_fill_for_grouping,
    cardinality_encode,
    apply_cardinality_encode_transductive,
)

Row = Mapping[Hashable, Hashable]


# --------------------------------------------------------------------------- #
# the only thing that changes: partition = equality on selected columns
# --------------------------------------------------------------------------- #
def projection_partition(
    rows: Sequence[Row],
    selected_cols: Sequence[Hashable],
    bin_fns: Optional[Mapping[Hashable, Callable[[Hashable], Hashable]]] = None,
) -> dict[int, Color]:
    """block_of[i] = hash of r_i restricted to selected_cols.

    selected_cols == []  ->  all rows in one block (the trivial / under-refined
    partition, i.e. S = empty). bin_fns optionally coarsens a column's values
    (Def. 3.3 binned incidence); default is identity (exact equality).
    """
    cols = list(selected_cols)
    block_of: dict[int, Color] = {}
    for i, r in enumerate(rows):
        if bin_fns:
            sig = tuple((r[c] if c not in bin_fns else bin_fns[c](r[c])) for c in cols)
        else:
            sig = tuple(r[c] for c in cols)
        block_of[i] = _stable_hash(sig)
    return block_of


# --------------------------------------------------------------------------- #
# main entry point — same signature shape as compute_J, minus the graph knobs
# --------------------------------------------------------------------------- #
def compute_J_incidence(
    rows: Sequence[Row],
    selected_cols: Sequence[Hashable],
    labels: Sequence[Label],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    lam: float = 1.0,
    loss: str = "log",
    bin_fns: Optional[Mapping[Hashable, Callable[[Hashable], Hashable]]] = None,
) -> JResult:
    """Compute J for the column-restricted incidence grable gamma_inc^S, without
    materialising the graph. rows is the FULL table (list of row mappings);
    selected_cols is S. labels is parallel to rows; train_idx/val_idx index into it.
    """
    block_of = projection_partition(rows, selected_cols, bin_fns)
    classes = sorted(set(labels))

    block_dist, prior = fit_block_predictor(block_of, labels, train_idx, classes)
    rv = val_risk(block_of, block_dist, prior, labels, val_idx, loss)
    om = occupancy(block_of, train_idx)

    return JResult(
        J=rv + lam * om,
        risk_val=rv,
        omega=om,
        lam=lam,
        num_blocks=len(set(block_of.values())),
        block_of=block_of,
    )


# --------------------------------------------------------------------------- #
# stand-alone, DataFrame-native entry point
# --------------------------------------------------------------------------- #
def compute_J_incidence_from_df(
    df_train: pd.DataFrame,
    target_col: str,
    selected_cols: Sequence[str],
    *,
    df_val: Optional[pd.DataFrame] = None,
    lam: float = 1.0,
    loss: str = "log",
    cardinality_encoding: bool = False,
    val_frac: float = 0.3,
    random_state: int = 0,
    bin_fns: Optional[Mapping[Hashable, Callable[[Hashable], Hashable]]] = None,
) -> JResult:
    """DataFrame-native wrapper around compute_J_incidence — no manual
    rows/labels/train_idx/val_idx plumbing required.

    df_train:   training DataFrame (always required).
    df_val:     optional external validation DataFrame. When omitted, a
                train/val split is drawn internally from df_train
                (val_frac / random_state), mirroring fit_autograble.
    cardinality_encoding:
                False (default) -> partition on the raw values in
                selected_cols (the "value dataframe").
                True  -> each selected column is first replaced by its
                value's frequency count (the "frequencies dataframe"), the
                same transform fit_autograble applies when its
                cardinality_encoding config flag is set: counts are fit on
                the train split ONLY (counts_train), then the val split is
                encoded transductively (counts over train UNION val), via
                utils.apply_cardinality_encode_transductive. This holds
                whether df_val is given explicitly or drawn from an internal
                split of df_train — the split always happens BEFORE encoding.
    lam, loss, bin_fns : forwarded to compute_J_incidence.

    Returns the same JResult as compute_J_incidence.
    """
    if target_col not in df_train.columns:
        raise ValueError(f"Target column {target_col!r} not found in df_train.columns")

    cols = list(selected_cols)
    df_train = df_train.copy()
    if df_val is not None:
        df_val = df_val.copy()

    if df_val is not None:
        df_tr, df_va = df_train, df_val
    else:
        n = len(df_train)
        tr_pos, va_pos = train_val_split(n, val_frac, random_state)
        df_tr = df_train.iloc[tr_pos].copy()
        df_va = df_train.iloc[va_pos].copy()

    if cardinality_encoding:
        df_tr, maps = cardinality_encode(df_tr, cols)
        df_va = apply_cardinality_encode_transductive(df_va, cols, maps)

    df_full = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    # NOTE: df_full[[]].to_dict("records") returns [] regardless of row count
    # (pandas can't infer row count from zero columns) — that would silently
    # drop every row and desync `rows` from train_idx/val_idx. S=empty is a
    # valid partition (single block, per Lemma 4.1), so keep one dict per row.
    if cols:
        rows = safe_fill_for_grouping(df_full[cols]).to_dict("records")
    else:
        rows = [{} for _ in range(len(df_full))]
    labels = df_full[target_col].tolist()
    train_idx = list(range(len(df_tr)))
    val_idx = list(range(len(df_tr), len(df_full)))

    return compute_J_incidence(
        rows,
        cols,
        labels,
        train_idx=train_idx,
        val_idx=val_idx,
        lam=lam,
        loss=loss,
        bin_fns=bin_fns,
    )


# --------------------------------------------------------------------------- #
# recommended fast path for a pandas table (big tables) — TODO: adopt this
# --------------------------------------------------------------------------- #
def projection_partition_pandas(df, selected_cols: Sequence[Hashable]):
    """Vectorised equality-projection via groupby. Returns block_of dict.

    GOTCHA: pandas groupby DROPS NaN groups by default. Pass dropna=False or the
    rows with a missing value in a selected column silently vanish from the
    partition (and from J). This matters for real relational tables with nulls.
    """
    cols = list(selected_cols)
    if not cols:
        return {i: 0 for i in range(len(df))}
    # ngroup gives a contiguous integer block id per row; dropna=False keeps nulls
    codes = df.groupby(cols, sort=False, dropna=False, observed=False).ngroup().to_numpy()
    return {i: int(codes[i]) for i in range(len(codes))}