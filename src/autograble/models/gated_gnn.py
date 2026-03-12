from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ..types import Stage2Config
from .base import BaseHeteroModel, _key


class HeteroGatedGNN(BaseHeteroModel):
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

    def forward(
        self,
        data: HeteroData,
        x_tab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
