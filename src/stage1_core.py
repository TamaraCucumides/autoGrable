from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .utils import omega_from_group_sizes, loss_from_proba


def build_block_probas(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build block-conditional class probabilities on train.

    Returns:
      block_probas: DataFrame indexed by block keys (MultiIndex if cols non-empty),
                    columns are class labels, values are probabilities.
      global_proba: Series of class priors.
    """
    classes = pd.Index(pd.unique(y_train))
    y_cat = pd.Categorical(y_train, categories=classes)

    # Attach y to X_train for groupby convenience
    tmp = X_train.copy()
    tmp["__y__"] = y_cat

    if cols:
        counts = tmp.groupby(cols, sort=False, dropna=False)["__y__"].value_counts().unstack(fill_value=0)
    else:
        # Single block
        counts = pd.DataFrame(
            [pd.Series(y_cat).value_counts().reindex(classes, fill_value=0).to_numpy()],
            index=pd.Index([()], name="__all__"),
            columns=classes,
        )

    # Ensure columns are exactly classes
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
    cols: List[str],
    block_probas: pd.DataFrame,
    global_proba: pd.Series,
) -> pd.DataFrame:
    """
    Row-wise lookup of block probabilities. Unseen blocks -> global prior.
    """
    classes = global_proba.index

    if not cols:
        return pd.DataFrame(
            np.tile(global_proba.to_numpy(), (len(X), 1)),
            index=X.index,
            columns=classes,
        )

    keys = pd.MultiIndex.from_frame(X[cols], names=cols)
    mapped = block_probas.reindex(keys)     # missing => NaN rows
    mapped = mapped.fillna(global_proba)    # fallback
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

    if cols:
        if omega_on == "train":
            gsz = X_train.groupby(cols, sort=False, dropna=False).size().to_numpy()
        else:
            gsz = X_val.groupby(cols, sort=False, dropna=False).size().to_numpy()
    else:
        gsz = np.array([len(X_train) if omega_on == "train" else len(X_val)], dtype=int)

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
    if omega_on not in ("train", "val"):
        raise ValueError("omega_on must be 'train' or 'val'")

    S = list(initial_cols)
    dropped: List[str] = []
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
        if len(S) <= 1:
            break

        best: Optional[Dict[str, Any]] = None
        best_drop: Optional[str] = None

        for c in S:
            cand_cols = [x for x in S if x != c]
            cand = evaluate_subset(
                X_train, y_train, X_val, y_val, cand_cols,
                lambda_=lambda_, loss_name=loss_name, omega_on=omega_on
            )
            if best is None or cand["J"] < best["J"]:
                best = cand
                best_drop = c

        assert best is not None and best_drop is not None
        improvement = float(cur["J"] - best["J"])

        history.append({
            "step": step + 1,
            "action": "drop" if improvement > min_improvement else "stop",
            "dropped": best_drop,
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
        dropped.append(best_drop)
        S = best["cols"]
        cur = best
        step += 1

    final_eval = cur
    return S, dropped, history, final_eval