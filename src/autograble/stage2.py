from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .types import Stage2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(col: str) -> str:
    """Sanitize column names for use as ModuleDict / ParameterDict keys."""
    return col.replace(".", "__dot__")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeteroGatedGNN(nn.Module):
    """
    Bipartite Gated GNN over a HeteroData graph from build_hetero_graph().

    Node types:
      "row"  — one learnable embedding per row
      <col>  — one learnable embedding per unique value in that column

    At each of num_layers steps:
      1. Gated messages flow row → value and value → row for every column.
      2. Each column has an independent learnable scalar gate (sigmoid-activated).
         Gates close to 0 mean that column contributes little structurally.
      3. Node states are updated with a GRU cell.

    Output: logits (classification) or scalars (regression) for every row node.
    """

    def __init__(
        self,
        cols: List[str],
        col_sizes: Dict[str, int],
        num_rows: int,
        num_out: int,
        config: Stage2Config,
        tab_dim: int = 0,
    ):
        super().__init__()
        H = config.hidden_dim
        self._cols = cols

        # Node embeddings (no input features needed)
        self.row_emb = nn.Embedding(num_rows, H)
        self.val_embs = nn.ModuleDict({
            _key(c): nn.Embedding(col_sizes[c], H) for c in cols
        })

        # Per-column scalar gates; init=0 → sigmoid(0)=0.5 at the start
        self.gates = nn.ParameterDict({
            _key(c): nn.Parameter(torch.zeros(1)) for c in cols
        })

        # Message linear layers (no bias to keep messages purely structural)
        self.msg_r2v = nn.ModuleDict({
            _key(c): nn.Linear(H, H, bias=False) for c in cols
        })
        self.msg_v2r = nn.ModuleDict({
            _key(c): nn.Linear(H, H, bias=False) for c in cols
        })

        # GRU cells for node state updates
        self.row_gru = nn.GRUCell(H, H)
        self.val_grus = nn.ModuleDict({
            _key(c): nn.GRUCell(H, H) for c in cols
        })

        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(H + tab_dim, num_out)
        self.num_layers = config.num_layers

    def forward(self, data: HeteroData, x_tab: Optional[torch.Tensor] = None) -> torch.Tensor:
        dev = next(self.parameters()).device

        h_row = self.row_emb(torch.arange(data["row"].num_nodes, device=dev))
        h_val = {
            c: self.val_embs[_key(c)](torch.arange(data[c].num_nodes, device=dev))
            for c in self._cols
        }

        for _ in range(self.num_layers):
            row_agg = torch.zeros_like(h_row)
            val_agg = {c: torch.zeros_like(h_val[c]) for c in self._cols}

            for c in self._cols:
                gate = torch.sigmoid(self.gates[_key(c)])  # scalar in (0, 1)
                ei = data["row", "has", c].edge_index.to(dev)
                src, dst = ei[0], ei[1]  # src=row node, dst=value node

                # Row → Value
                m = self.msg_r2v[_key(c)](h_row[src]) * gate
                val_agg[c].scatter_add_(0, dst.unsqueeze(1).expand_as(m), m)

                # Value → Row
                m = self.msg_v2r[_key(c)](h_val[c][dst]) * gate
                row_agg.scatter_add_(0, src.unsqueeze(1).expand_as(m), m)

            h_row = self.row_gru(self.dropout(row_agg), h_row)
            for c in self._cols:
                h_val[c] = self.val_grus[_key(c)](self.dropout(val_agg[c]), h_val[c])

        h = torch.cat([h_row, x_tab], dim=1) if x_tab is not None else h_row
        return self.head(self.dropout(h))  # [num_rows, num_out]

    def gate_values(self) -> Dict[str, float]:
        """Return sigmoid(gate) for each column — closer to 1 means more relevant."""
        return {c: float(torch.sigmoid(self.gates[_key(c)]).item()) for c in self._cols}


# ---------------------------------------------------------------------------
# Gate inspection
# ---------------------------------------------------------------------------

def gate_summary(result: "Stage2Result", threshold: float = 0.2) -> pd.DataFrame:
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
        result:    Stage2Result from fit_gated_gnn()
        threshold: gate value below which a column is labelled "ignored" (default 0.2)
    """
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
# Result
# ---------------------------------------------------------------------------

@dataclass
class Stage2Result:
    model: HeteroGatedGNN
    edge_gates: Dict[str, float]    # col -> gate value in (0, 1) after training
    train_losses: List[float]
    val_losses: Optional[List[float]]
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_gated_gnn(
    graph: HeteroData,
    y: torch.Tensor,
    config: Stage2Config,
    x_tab: Optional[torch.Tensor] = None,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
) -> Stage2Result:
    """
    Train a HeteroGatedGNN on the given graph.

    Args:
        graph:      HeteroData from build_hetero_graph()
        y:          Labels for row nodes. Long tensor for classification,
                    float tensor for regression.
        config:     Stage2Config
        x_tab:      Optional raw tabular features [num_rows, D]. Concatenated
                    with the GNN's structural embedding before the output head,
                    so the GNN is free to focus purely on graph structure.
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

    model = HeteroGatedGNN(
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
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "lr": config.lr,
            "epochs": config.epochs,
            "task": config.task,
            "device": config.device,
        },
    )
