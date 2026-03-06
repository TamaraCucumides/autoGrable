from __future__ import annotations

from typing import Optional, List
import pandas as pd

from .types import Stage1Config, Stage1Result
from .candidates import select_candidate_columns
from .utils import train_val_split, safe_fill_for_grouping, cardinality_encode
from .stage1_core import greedy_backward_elimination, build_block_probas


def fit_structure_stage1(df: pd.DataFrame, config: Stage1Config) -> Stage1Result:
    """
    Stage 1 (classification): greedy backward elimination over structural columns.

    - Induces a partition via equality on selected columns S
    - Learns block-conditional class probabilities on TRAIN
    - Minimizes J(S) = val_loss + lambda * Omega(sample, pi)
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
    )

    # Preprocessing: replace candidate column values with their frequency count in df
    cardinality_maps = None
    if config.cardinality_encoding:
        df, cardinality_maps = cardinality_encode(df, eligible_cols)

    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 rows to run Stage 1.")

    train_idx, val_idx = train_val_split(n, config.val_frac, config.random_state)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    y_train = df_train[y_col].copy()
    y_val = df_val[y_col].copy()

    # grouping-safe views for eligible cols only
    X_train = safe_fill_for_grouping(df_train[eligible_cols])
    X_val = safe_fill_for_grouping(df_val[eligible_cols])

    selected_cols, dropped_cols, history, _final_eval = greedy_backward_elimination(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        initial_cols=eligible_cols,
        lambda_=config.lambda_,
        loss_name=config.loss_name,
        omega_on=config.omega_on,
        min_improvement=config.min_improvement,
        max_steps=config.max_steps,
    )

    # Fit final block model on TRAIN for selected columns
    block_probas, global_proba = build_block_probas(X_train, y_train, selected_cols)

    return Stage1Result(
        selected_cols=selected_cols,
        dropped_cols=dropped_cols,
        history=history,
        excluded_by_rule=excluded_by_rule,
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
        },
        cardinality_maps=cardinality_maps,
    )