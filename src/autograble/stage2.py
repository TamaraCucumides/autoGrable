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
    graph: HeteroData,
    y: torch.Tensor,
    config: Stage2Config,
    model_cls: Type[BaseHeteroModel] = HeteroGatedGNN,
    x_tab: Optional[torch.Tensor] = None,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
) -> Stage2Result:
    """
    Build and train any BaseHeteroModel on the given graph.

    Args:
        graph:      HeteroData from build_hetero_graph()
        y:          Labels for row nodes. Long tensor for classification,
                    float tensor for regression.
        config:     Stage2Config
        model_cls:  Model class to instantiate (default: HeteroGatedGNN).
                    Must follow the BaseHeteroModel constructor signature:
                    (cols, col_sizes, num_rows, num_out, config, tab_dim).
        x_tab:      Optional raw tabular features [num_rows, D]. Concatenated
                    with the GNN embedding before the output head.
        train_mask: Boolean mask over row nodes for training (default: all rows)
        val_mask:   Boolean mask over row nodes for validation (optional)
    """
    dev = torch.device(config.device)
    n = graph["row"].num_nodes

    if train_mask is None:
        train_mask = torch.ones(n, dtype=torch.bool)

    y = y.to(dev)
    train_mask = train_mask.to(dev)
    val_mask = val_mask.to(dev) if val_mask is not None else None
    x_tab = x_tab.to(dev) if x_tab is not None else None

    cols = [t for t in graph.node_types if t != "row"]
    col_sizes = {c: graph[c].num_nodes for c in cols}

    if config.task == "classification":
        num_out = int(y.max().item()) + 1
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        num_out = 1
        loss_fn = nn.MSELoss()

    model = model_cls(
        cols=cols,
        col_sizes=col_sizes,
        num_rows=n,
        num_out=num_out,
        config=config,
        tab_dim=x_tab.shape[1] if x_tab is not None else 0,
    ).to(dev)

    graph = graph.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for _ in range(config.epochs):
        model.train()
        opt.zero_grad()
        out = model(graph, x_tab)
        target = y[train_mask]
        pred = out[train_mask] if config.task == "classification" else out[train_mask].squeeze()
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                out = model(graph, x_tab)
                target = y[val_mask]
                pred = out[val_mask] if config.task == "classification" else out[val_mask].squeeze()
                val_losses.append(float(loss_fn(pred, target).item()))

    return Stage2Result(
        model=model,
        edge_gates=model.gate_values(),
        train_losses=train_losses,
        val_losses=val_losses if val_mask is not None else None,
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
    graph: HeteroData,
    y: torch.Tensor,
    config: Stage2Config,
    x_tab: Optional[torch.Tensor] = None,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
) -> Stage2Result:
    """Convenience wrapper for fit_stage2 with model_cls=HeteroGatedGNN."""
    return fit_stage2(
        graph, y, config,
        model_cls=HeteroGatedGNN,
        x_tab=x_tab,
        train_mask=train_mask,
        val_mask=val_mask,
    )
