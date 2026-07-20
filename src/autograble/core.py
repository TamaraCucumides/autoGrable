from __future__ import annotations

from typing import Optional, List
import pandas as pd

from .types import AutoGrableConfig, AutoGrableResult
from .candidates import select_candidate_columns
from .utils import (
    train_val_split,
    safe_fill_for_grouping,
    cardinality_encode,
    apply_cardinality_encode_transductive,
)
from .selection import greedy_selection, build_block_probas


def fit_autograble(
    df: pd.DataFrame,
    config: AutoGrableConfig,
    df_val: Optional[pd.DataFrame] = None,
) -> AutoGrableResult:
    """
    autoGrable (classification): greedy structural column selection.

    - Induces a partition via equality on selected columns S
    - Learns block-conditional class probabilities on TRAIN
    - Minimizes J(S) = val_loss + lambda * Omega(sample, pi)
    - config.direction picks the search strategy:
        "backward" (default): start from all columns selected and greedily
            drop the column whose removal improves J the most.
        "forward": start from 0 columns selected (single block) and
            greedily add the column whose inclusion improves J the most.

    Args:
        df:     Training DataFrame (always required).
        config: AutoGrableConfig.
        df_val: Optional external validation DataFrame. When provided, the
                internal train/val split (config.val_frac) is skipped and
                df_val is used directly for lambda selection. Cardinality
                maps are computed from df (train) alone; df_val is then
                encoded transductively, with counts over df UNION df_val.
    """
    y_col = config.y_col
    if y_col not in df.columns:
        raise ValueError(f"Target column {y_col!r} not found in df.columns")

    eligible_cols, excluded_by_rule = select_candidate_columns(
        df,
        y_col=y_col,
        candidate_cols=config.candidate_cols,
        force_include=config.force_include,
        force_exclude=config.force_exclude,
        name_hint_exclude=() if not config.drop_key_like_cols else (
            "id", "uuid", "guid", "hash", "session", "token"
        ),
        exclude_near_unique=config.drop_near_unique_cols,
        unique_frac_threshold=config.near_unique_threshold,
        exclude_high_cardinality_object=config.drop_high_cardinality_object_cols,
        object_unique_frac_threshold=config.object_unique_frac_threshold,
        exclude_datetimes=config.drop_datetime_cols,
    )

    # Cardinality encoding: df (train) is encoded from its own frequencies.
    # df_val, when given, is encoded transductively — each value's count
    # includes its occurrences in both df (train) and df_val.
    cardinality_maps = None
    if config.cardinality_encoding:
        df, cardinality_maps = cardinality_encode(df, eligible_cols)
        if df_val is not None:
            df_val = apply_cardinality_encode_transductive(df_val, eligible_cols, cardinality_maps)

    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 rows to run autoGrable.")

    if df_val is not None:
        df_train = df
        df_val_split = df_val
    else:
        train_idx, val_idx = train_val_split(n, config.val_frac, config.random_state)
        df_train = df.iloc[train_idx].copy()
        df_val_split = df.iloc[val_idx].copy()

    y_train = df_train[y_col].copy()
    y_val = df_val_split[y_col].copy()

    # grouping-safe views for eligible cols only
    X_train = safe_fill_for_grouping(df_train[eligible_cols])
    X_val = safe_fill_for_grouping(df_val_split[eligible_cols])

    selected_cols, dropped_cols, history, final_eval = greedy_selection(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        initial_cols=eligible_cols,
        direction=config.direction,
        lambda_=config.lambda_,
        loss_name=config.loss_name,
        omega_on=config.omega_on,
        min_improvement=config.min_improvement,
        max_steps=config.max_steps,
    )

    # Fit final block model on TRAIN for selected columns
    block_probas, global_proba = build_block_probas(X_train, y_train, selected_cols)

    return AutoGrableResult(
        selected_cols=selected_cols,
        dropped_cols=dropped_cols,
        history=history,
        excluded_by_rule=excluded_by_rule,
        final_J=final_eval["J"],
        final_val_loss=final_eval["val_loss"],
        final_omega=final_eval["omega"],
        block_probas=block_probas,
        global_proba=global_proba,
        config={
            "y_col": config.y_col,
            "candidate_cols": config.candidate_cols,
            "force_include": config.force_include,
            "force_exclude": config.force_exclude,
            "val_frac": config.val_frac,
            "random_state": config.random_state,
            "lambda_": config.lambda_,
            "loss_name": config.loss_name,
            "min_improvement": config.min_improvement,
            "max_steps": config.max_steps,
            "omega_on": config.omega_on,
            "cardinality_encoding": config.cardinality_encoding,
            "direction": config.direction,
            "drop_key_like_cols": config.drop_key_like_cols,
            "drop_near_unique_cols": config.drop_near_unique_cols,
            "near_unique_threshold": config.near_unique_threshold,
            "drop_high_cardinality_object_cols": config.drop_high_cardinality_object_cols,
            "object_unique_frac_threshold": config.object_unique_frac_threshold,
            "drop_datetime_cols": config.drop_datetime_cols,
        },
        cardinality_maps=cardinality_maps,
    )
