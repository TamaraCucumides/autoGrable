from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .types import Stage2Config
from .models.base import BaseHeteroModel
from .models.gated_gnn import HeteroGatedGNN


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Stage2Result:
    model: BaseHeteroModel
    edge_gates: Dict[str, float]    # col -> gate value in (0, 1); {} if model has no gates
    train_losses: List[float]
    val_losses: Optional[List[float]]
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Gate inspection
# ---------------------------------------------------------------------------

def gate_summary(result: Stage2Result, threshold: float = 0.2) -> pd.DataFrame:
    """
    Return a DataFrame summarising each column's learned gate value, sorted
    from most ignored (gate → 0) to most active (gate → 1).

    Columns:
      column   — column name
      gate     — sigmoid(learned scalar), in (0, 1)
      status   — "ignored"  if gate < threshold
                 "weak"     if threshold <= gate < 0.5
                 "active"   if gate >= 0.5

    Args:
        result:    Stage2Result from fit_stage2() or fit_gated_gnn()
        threshold: gate value below which a column is labelled "ignored" (default 0.2)
    """
    if not result.edge_gates:
        raise ValueError("This model does not expose gate values.")

    def _status(g: float) -> str:
        if g < threshold:
            return "ignored"
        if g < 0.5:
            return "weak"
        return "active"

    rows = [
        {"column": col, "gate": round(gate, 4), "status": _status(gate)}
        for col, gate in result.edge_gates.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("gate", ascending=True)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_stage2(
    graph_train: HeteroData,
    y_train: torch.Tensor,
    config: Stage2Config,
    model_cls: Type[BaseHeteroModel] = HeteroGatedGNN,
    x_tab_train: Optional[torch.Tensor] = None,
    graph_val: Optional[HeteroData] = None,
    y_val: Optional[torch.Tensor] = None,
    x_tab_val: Optional[torch.Tensor] = None,
) -> Stage2Result:
    """
    Build and train any BaseHeteroModel in an inductive fashion.

    Each split is a separate graph — no masks. The model generalises to
    val/test graphs because it uses feature-based (not ID-based) node
    initialisation (see HeteroGatedGNN).

    Args:
        graph_train:   HeteroData for the training split.
        y_train:       Labels for all row nodes in graph_train.
                       Long tensor for classification, float for regression.
        config:        Stage2Config.
        model_cls:     Model class to instantiate (default: HeteroGatedGNN).
        x_tab_train:   Optional tabular features [n_train, D] for the head.
        graph_val:     Optional HeteroData for the validation split.
        y_val:         Labels for all row nodes in graph_val.
        x_tab_val:     Optional tabular features [n_val, D] for the head.
    """
    dev = torch.device(config.device)

    y_train = y_train.to(dev)
    x_tab_train = x_tab_train.to(dev) if x_tab_train is not None else None

    if graph_val is not None:
        assert y_val is not None, "y_val required when graph_val is provided"
        y_val = y_val.to(dev)
        x_tab_val = x_tab_val.to(dev) if x_tab_val is not None else None

    cols = [t for t in graph_train.node_types if t != "row"]
    col_sizes = {c: graph_train[c].num_nodes for c in cols}

    if config.task == "classification":
        num_out = int(y_train.max().item()) + 1
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        num_out = 1
        loss_fn = nn.MSELoss()

    row_x = graph_train["row"].get("x")
    row_feat_dim = row_x.shape[1] if row_x is not None else 0

    model = model_cls(
        cols=cols,
        col_sizes=col_sizes,
        num_rows=graph_train["row"].num_nodes,
        num_out=num_out,
        config=config,
        tab_dim=x_tab_train.shape[1] if x_tab_train is not None else 0,
        row_feat_dim=row_feat_dim,
    ).to(dev)

    graph_train = graph_train.to(dev)
    if graph_val is not None:
        graph_val = graph_val.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for _ in range(config.epochs):
        model.train()
        opt.zero_grad()
        out = model(graph_train, x_tab_train)
        pred = out if config.task == "classification" else out.squeeze()
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        if graph_val is not None:
            model.eval()
            with torch.no_grad():
                out_val = model(graph_val, x_tab_val)
                pred_val = out_val if config.task == "classification" else out_val.squeeze()
                val_losses.append(float(loss_fn(pred_val, y_val).item()))

    return Stage2Result(
        model=model,
        edge_gates=model.gate_values(),
        train_losses=train_losses,
        val_losses=val_losses if graph_val is not None else None,
        config={
            "model": model_cls.__name__,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "lr": config.lr,
            "epochs": config.epochs,
            "task": config.task,
            "device": config.device,
        },
    )


def fit_gated_gnn(
    graph_train: HeteroData,
    y_train: torch.Tensor,
    config: Stage2Config,
    x_tab_train: Optional[torch.Tensor] = None,
    graph_val: Optional[HeteroData] = None,
    y_val: Optional[torch.Tensor] = None,
    x_tab_val: Optional[torch.Tensor] = None,
) -> Stage2Result:
    """Convenience wrapper for fit_stage2 with model_cls=HeteroGatedGNN."""
    return fit_stage2(
        graph_train, y_train, config,
        model_cls=HeteroGatedGNN,
        x_tab_train=x_tab_train,
        graph_val=graph_val,
        y_val=y_val,
        x_tab_val=x_tab_val,
    )
