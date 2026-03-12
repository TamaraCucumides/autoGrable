from .base import BaseHeteroModel
from .gated_gnn import HeteroGatedGNN

# Registry: maps model name → class.
# Add new models here as they are implemented.
MODELS: dict = {
    "gated_gnn": HeteroGatedGNN,
}

__all__ = ["BaseHeteroModel", "HeteroGatedGNN", "MODELS"]
