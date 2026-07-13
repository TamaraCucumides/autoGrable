from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .utils import omega_from_group_sizes, loss_from_proba


def build_block_probas(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Block-conditional class probabilities estimated on train.
    Uses hashed block keys to avoid MultiIndex cartesian blow-ups.
    """
    classes = pd.Index(pd.unique(y_train))
    y_cat = pd.Categorical(y_train, categories=classes)

    key = make_block_key(X_train, cols)

    # counts: rows=blocks, cols=classes
    counts = pd.crosstab(key, y_cat, dropna=False)

    # Ensure all classes are columns (in case a class is missing in train split)
    for cl in classes:
        if cl not in counts.columns:
            counts[cl] = 0
    counts = counts[classes]

    totals = counts.sum(axis=1).replace(0, 1)
    block_probas = counts.div(totals, axis=0)

    global_counts = pd.Series(y_cat).value_counts().reindex(classes, fill_value=0)
    global_proba = global_counts / max(int(global_counts.sum()), 1)

    return block_probas, global_proba


def predict_proba_from_blocks(
    X: pd.DataFrame,
    cols: list[str],
    block_probas: pd.DataFrame,
    global_proba: pd.Series,
) -> pd.DataFrame:
    """
    Lookup block probabilities using hashed keys; unseen blocks fall back to global prior.
    """
    classes = global_proba.index
    key = make_block_key(X, cols)

    mapped = block_probas.reindex(key.values)   # index of block_probas is block key
    mapped = mapped.fillna(global_proba)
    mapped.index = X.index
    mapped.columns = classes
    return mapped


def evaluate_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cols: List[str],
    *,
    lambda_: float,
    loss_name: str,
    omega_on: str,  # "train" or "val"
) -> Dict[str, Any]:
    block_probas, global_proba = build_block_probas(X_train, y_train, cols)

    proba_val = predict_proba_from_blocks(X_val, cols, block_probas, global_proba)
    val_loss = loss_from_proba(y_val, proba_val, loss_name=loss_name)

    if omega_on == "train":
        key_for_omega = make_block_key(X_train, cols)
    else:
        key_for_omega = make_block_key(X_val, cols)

    gsz = key_for_omega.value_counts().to_numpy()

    omega = omega_from_group_sizes(gsz)
    J = float(val_loss + lambda_ * omega)

    return {
        "cols": cols,
        "J": J,
        "val_loss": float(val_loss),
        "omega": float(omega),
        "n_blocks_train": int(block_probas.shape[0]) if cols else 1,
        "block_probas": block_probas,
        "global_proba": global_proba,
    }


def greedy_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    initial_cols: List[str],
    *,
    direction: str = "backward",
    lambda_: float,
    loss_name: str,
    omega_on: str = "train",
    min_improvement: float = 0.0,
    max_steps: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Greedy structural column selection, minimizing J(S) = val_loss + lambda * Omega(S).

    direction="backward": start with S = all of initial_cols (every column
        distinguishes a block) and greedily drop the column whose removal
        improves J the most, stopping when no drop improves J (by more than
        min_improvement) or when a single column remains.

    direction="forward": start with S = [] (all rows are a single block)
        and greedily add the column whose inclusion improves J the most,
        stopping when no addition improves J or every column has been added.

    Returns:
      selected_cols, unselected_cols, history, final_eval

      unselected_cols is the columns dropped (backward) or never added
      (forward).
    """
    if direction not in ("backward", "forward"):
        raise ValueError("direction must be 'backward' or 'forward'")
    if omega_on not in ("train", "val"):
        raise ValueError("omega_on must be 'train' or 'val'")

    backward = direction == "backward"

    S: List[str] = list(initial_cols) if backward else []
    pool: List[str] = [] if backward else list(initial_cols)
    unselected: List[str] = []
    history: List[Dict[str, Any]] = []

    cur = evaluate_subset(
        X_train, y_train, X_val, y_val, S,
        lambda_=lambda_, loss_name=loss_name, omega_on=omega_on
    )
    history.append({
        "step": 0,
        "action": "init",
        "cols": S.copy(),
        "J": cur["J"],
        "val_loss": cur["val_loss"],
        "omega": cur["omega"],
        "n_blocks_train": cur["n_blocks_train"],
    })

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        candidates = S if backward else pool
        if backward and len(S) <= 1:
            break
        if not backward and not pool:
            break

        best: Optional[Dict[str, Any]] = None
        best_move: Optional[str] = None

        for c in candidates:
            cand_cols = [x for x in S if x != c] if backward else S + [c]
            cand = evaluate_subset(
                X_train, y_train, X_val, y_val, cand_cols,
                lambda_=lambda_, loss_name=loss_name, omega_on=omega_on
            )
            if best is None or cand["J"] < best["J"]:
                best = cand
                best_move = c

        assert best is not None and best_move is not None
        improvement = float(cur["J"] - best["J"])
        move_key = "dropped" if backward else "added"
        move_action = "drop" if backward else "add"

        history.append({
            "step": step + 1,
            "action": move_action if improvement > min_improvement else "stop",
            move_key: best_move,
            "improvement": improvement,
            "cols": best["cols"].copy(),
            "J": best["J"],
            "val_loss": best["val_loss"],
            "omega": best["omega"],
            "n_blocks_train": best["n_blocks_train"],
        })

        if improvement <= min_improvement:
            break

        # accept
        if backward:
            unselected.append(best_move)
        else:
            pool = [x for x in pool if x != best_move]
        S = best["cols"]
        cur = best
        step += 1

    if not backward:
        unselected = [c for c in initial_cols if c not in S]

    final_eval = cur
    return S, unselected, history, final_eval


def greedy_backward_elimination(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    initial_cols: List[str],
    *,
    lambda_: float,
    loss_name: str,
    omega_on: str = "train",
    min_improvement: float = 0.0,
    max_steps: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      selected_cols, dropped_cols, history, final_eval
    """
    return greedy_selection(
        X_train, y_train, X_val, y_val, initial_cols,
        direction="backward",
        lambda_=lambda_, loss_name=loss_name, omega_on=omega_on,
        min_improvement=min_improvement, max_steps=max_steps,
    )


def greedy_forward_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    initial_cols: List[str],
    *,
    lambda_: float,
    loss_name: str,
    omega_on: str = "train",
    min_improvement: float = 0.0,
    max_steps: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      selected_cols, unselected_cols, history, final_eval
    """
    return greedy_selection(
        X_train, y_train, X_val, y_val, initial_cols,
        direction="forward",
        lambda_=lambda_, loss_name=loss_name, omega_on=omega_on,
        min_improvement=min_improvement, max_steps=max_steps,
    )


def make_block_key(X: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Stable-ish block id for each row based on equality of values in `cols`.
    Requires that X has already been 'safe-filled' (no NaN != NaN surprises).
    """
    if not cols:
        return pd.Series(np.zeros(len(X), dtype="uint64"), index=X.index, name="__block__")

    # Hash the selected columns row-wise
    # index=False: ignore original index in the hash
    key = pd.util.hash_pandas_object(X[cols], index=False).astype("uint64")
    key.name = "__block__"
    key.index = X.index
    return key