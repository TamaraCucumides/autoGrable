from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ..types import Stage2Config


def _key(col: str) -> str:
    """Sanitize column names for use as ModuleDict / ParameterDict keys."""
    return col.replace(".", "__dot__")


class BaseHeteroModel(nn.Module, ABC):
    """
    Interface all Stage-2 models must satisfy.

    Constructor signature (enforced by fit_stage2):
        __init__(cols, col_sizes, num_rows, num_out, config, tab_dim)

    Subclasses must implement forward() and may override gate_values().
    """

    @abstractmethod
    def forward(
        self,
        data: HeteroData,
        x_tab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            data:  HeteroData from build_hetero_graph()
            x_tab: optional tabular features [num_rows, D]
        Returns:
            logits / scalars [num_rows, num_out]
        """

    def gate_values(self) -> Dict[str, float]:
        """
        Return a col -> gate_value dict for interpretability.
        Models without explicit gates should return an empty dict.
        """
        return {}
