from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch


def make_tabular_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    numeric_fill: str = "mean",
    encode_categoricals: bool = True,
) -> torch.Tensor:
    """
    Preprocess a DataFrame into a float tensor for use as x_tab in fit_gated_gnn().

    Processing per column type:
      - Numeric / boolean : NaN filled with column mean (or 0), kept as-is.
      - Categorical / object: one-hot encoded (if encode_categoricals=True),
                               NaN becomes its own "__NaN__" category.
      - Everything else (datetime, etc.): skipped.

    Args:
        df:                   Input DataFrame.
        exclude_cols:         Columns to drop before processing — typically the
                              target column and the Stage-1 selected columns
                              (those are already represented in the graph).
        numeric_fill:         NaN strategy for numeric columns: "mean" or "zero".
        encode_categoricals:  One-hot encode object / categorical columns.

    Returns:
        Float tensor of shape [num_rows, num_features].
    """
    df = df.copy()

    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    parts: List[pd.DataFrame] = []

    for col in df.columns:
        s = df[col]

        # Boolean → float
        if pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(float).to_frame())
            continue

        # Numeric → fill NaN, keep
        if pd.api.types.is_numeric_dtype(s):
            fill = s.mean() if numeric_fill == "mean" else 0.0
            parts.append(s.fillna(fill).to_frame())
            continue

        # Categorical / object → one-hot
        if encode_categoricals and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            s = s.astype(str).where(s.notna(), "__NaN__")
            dummies = pd.get_dummies(s, prefix=col, dtype=float)
            parts.append(dummies)
            continue

        # All other dtypes (datetime, etc.) are skipped

    if not parts:
        raise ValueError(
            "No usable columns found after filtering. "
            "Check exclude_cols and the dtypes of your DataFrame."
        )

    X = pd.concat(parts, axis=1).astype(float)
    return torch.tensor(X.values, dtype=torch.float)
