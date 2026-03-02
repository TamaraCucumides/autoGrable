from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def select_candidate_columns(
    df: pd.DataFrame,
    *,
    y_col: str,
    candidate_cols: Optional[Sequence[str]] = None,
    force_include: Optional[Sequence[str]] = None,
    force_exclude: Optional[Sequence[str]] = None,
    unique_frac_threshold: float = 0.98,
    object_unique_frac_threshold: float = 0.80,
    exclude_datetimes: bool = True,
    exclude_high_cardinality_object: bool = True,
    name_hint_exclude: Tuple[str, ...] = ("id", "uuid", "guid", "hash", "session", "token"),
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
      eligible_cols: columns allowed to participate in Stage-1 structure
      excluded_by_rule: audit trail of what got excluded and why

    If candidate_cols is provided: it's the initial eligible set (minus rules/overrides).
    If candidate_cols is None: initial set is all df columns except y_col.
    """
    if y_col not in df.columns:
        raise ValueError(f"y_col={y_col!r} not in df.columns")

    base_cols = list(candidate_cols) if candidate_cols is not None else [c for c in df.columns if c != y_col]
    force_include = list(force_include) if force_include is not None else []
    force_exclude = list(force_exclude) if force_exclude is not None else []

    excluded: Dict[str, List[str]] = {}

    # Ensure target not eligible
    if y_col in base_cols:
        base_cols.remove(y_col)
        excluded.setdefault("target", []).append(y_col)

    n = len(df)
    if n == 0:
        # degenerate, but keep user intent
        eligible = [c for c in base_cols if c not in force_exclude] + [c for c in force_include if c != y_col]
        # de-dup preserving order
        seen = set()
        eligible_out = []
        for c in eligible:
            if c not in seen and c in df.columns and c != y_col:
                seen.add(c)
                eligible_out.append(c)
        return eligible_out, excluded

    # Rule: name-hints (often IDs)
    for c in list(base_cols):
        lc = str(c).lower()
        if any(h in lc for h in name_hint_exclude):
            base_cols.remove(c)
            excluded.setdefault("name_hint", []).append(c)

    # Rule: near-unique columns
    for c in list(base_cols):
        nun = df[c].nunique(dropna=False)
        if nun / max(n, 1) >= unique_frac_threshold:
            base_cols.remove(c)
            excluded.setdefault("near_unique", []).append(c)

    # Rule: datetime columns (often near-unique)

    if exclude_datetimes:
        for c in list(base_cols):
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                base_cols.remove(c)
                excluded.setdefault("datetime", []).append(c)

    # Rule: high-cardinality object columns
    if exclude_high_cardinality_object:
        for c in list(base_cols):
            if df[c].dtype == "object":
                nun = df[c].nunique(dropna=False)
                if nun / max(n, 1) >= object_unique_frac_threshold:
                    base_cols.remove(c)
                    excluded.setdefault("high_cardinality_object", []).append(c)

    # Apply user overrides
    eligible_set = set(base_cols)
    eligible_set |= set(force_include)
    eligible_set -= set(force_exclude)
    eligible_set.discard(y_col)

    # Preserve a stable order:
    # - If candidate_cols given: preserve that order
    # - Else: preserve df column order
    if candidate_cols is not None:
        ordered = [c for c in candidate_cols if c in eligible_set and c != y_col]
    else:
        ordered = [c for c in df.columns if c in eligible_set and c != y_col]

    if not ordered:
        raise ValueError("No eligible structural columns after filtering/overrides.")

    return ordered, excluded