from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class Stage1Config:
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


@dataclass
class Stage1Result:
    selected_cols: List[str]
    dropped_cols: List[str]
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
class Stage2Config:
    hidden_dim: int = 64
    num_layers: int = 3           # GRU unroll steps
    dropout: float = 0.0
    lr: float = 1e-3
    epochs: int = 200
    task: str = "classification"  # "classification" or "regression"
    device: str = "cpu"