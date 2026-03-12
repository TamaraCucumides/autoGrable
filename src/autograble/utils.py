from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def train_val_split(n: int, val_frac: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0,1)")
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(np.floor(val_frac * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def safe_fill_for_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make equality/grouping deterministic:
    - Replace missing values with stable sentinels.
    - Handle pandas Categoricals explicitly.
    """
    out = df.copy()

    for c in out.columns:
        s = out[c]

        # --- IMPORTANT: handle categoricals first ---
        if pd.api.types.is_categorical_dtype(s):
            out[c] = s.astype("object").where(s.notna(), "__MISSING__")
            continue

        # datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.fillna(pd.Timestamp.min)
            continue

        # numeric / boolean
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = s.fillna(-10**18)
            continue

        # fallback: object / string
        out[c] = s.astype("object").where(s.notna(), "__MISSING__")

    return out


def cardinality_encode(
    df: pd.DataFrame, cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Replace each value in cols with its frequency count in df.

    For example, if 'City' has ['Paris', 'Paris', 'Lyon'], it becomes [2, 2, 1].
    NaN values are counted as their own group.

    Returns:
      df_encoded: copy of df with cols replaced by integer counts
      maps: dict of col -> value_counts Series (for applying to new data later)
    """
    df = df.copy()
    maps: Dict[str, pd.Series] = {}
    for col in cols:
        maps[col] = df[col].value_counts(dropna=False)
        df[col] = df.groupby(col, dropna=False)[col].transform("size").astype(int)
    return df, maps


def apply_cardinality_encode(
    df: pd.DataFrame, cols: List[str], maps: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Apply pre-computed cardinality maps (from cardinality_encode) to a new df.

    Values not seen in the training maps are assigned count 0.
    """
    df = df.copy()
    for col in cols:
        counts = maps[col]  # value -> count Series from train
        nan_count = int(counts.get(float("nan"), counts.get(None, 0)))
        def _map(v):
            if pd.isna(v):
                return nan_count
            return int(counts.get(v, 0))
        df[col] = df[col].map(_map).astype(int)
    return df


def omega_from_group_sizes(group_sizes: np.ndarray) -> float:
    n = float(group_sizes.sum())
    if n <= 0:
        return 0.0
    return float(np.sqrt(group_sizes[group_sizes > 0]).sum() / n)


def loss_from_proba(
    y_true: pd.Series,
    proba: pd.DataFrame,
    loss_name: str = "logloss",
    eps: float = 1e-12,
) -> float:
    """
    y_true: series of labels
    proba: DataFrame (n, n_classes) with same index order as y_true, columns are classes.
    """
    classes = proba.columns
    y_cat = pd.Categorical(y_true, categories=classes)
    y_idx = y_cat.codes  # -1 if unseen label
    P = proba.to_numpy()

    if loss_name == "logloss":
        P = np.clip(P, eps, 1.0)
        bad = (y_idx < 0)
        ll = np.zeros(len(y_true), dtype=float)
        ll[~bad] = -np.log(P[np.arange(len(y_true))[~bad], y_idx[~bad]])
        ll[bad] = -np.log(1.0 / P.shape[1])
        return float(ll.mean())

    if loss_name in ("0-1", "01", "error"):
        y_hat = classes.to_numpy()[np.argmax(P, axis=1)]
        return float((y_hat != y_true.to_numpy()).mean())

    raise ValueError(f"Unknown loss_name={loss_name!r}. Use 'logloss' or '0-1'.")