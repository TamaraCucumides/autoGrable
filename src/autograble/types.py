from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class AutoGrableConfig:
    y_col: str
    candidate_cols: Optional[List[str]] = None
    force_include: Optional[List[str]] = None
    force_exclude: Optional[List[str]] = None

    val_frac: float = 0.2
    random_state: int = 0

    lambda_: float = 1.0
    loss_name: str = "logloss"   # "logloss" or "0-1"
    min_improvement: float = 0.0
    max_steps: Optional[int] = None

    omega_on: str = "train"      # "train" (recommended) or "val"
    cardinality_encoding: bool = False

    drop_key_like_cols: bool = True  # exclude id/uuid/guid/hash/session/token-like
                                      # columns during candidate selection

    drop_near_unique_cols: bool = True   # exclude near-unique columns (likely keys)
    near_unique_threshold: float = 0.98  # frac of distinct values that counts as "near-unique"

    drop_high_cardinality_object_cols: bool = True   # exclude high-cardinality object/text cols
    object_unique_frac_threshold: float = 0.80       # frac of distinct values that triggers it

    drop_datetime_cols: bool = True  # exclude datetime-dtype columns

    direction: str = "backward"  # "backward" (start from all cols, drop) or
                                  # "forward" (start from 0 cols, add)


@dataclass
class AutoGrableResult:
    selected_cols: List[str]
    dropped_cols: List[str]  # backward: cols removed; forward: cols never added
    history: List[Dict[str, Any]]
    excluded_by_rule: Dict[str, List[str]]

    # learned block model on TRAIN for the final selected columns
    block_probas: pd.DataFrame  # index = block key (MultiIndex), columns = classes, values = probs
    global_proba: pd.Series     # class priors on train

    config: Dict[str, Any]

    # cardinality encoding maps: col -> value_counts Series (value -> count in original df)
    # None when cardinality_encoding=False
    cardinality_maps: Optional[Dict[str, pd.Series]] = None


@dataclass(frozen=True)
class RefinementConfig:
    hidden_dim: int = 64
    num_layers: int = 3           # GRU unroll steps
    dropout: float = 0.0
    lr: float = 1e-3
    epochs: int = 200
    task: str = "classification"  # "classification" or "regression"
    device: str = "cpu"
