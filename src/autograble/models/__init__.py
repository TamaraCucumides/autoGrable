from .base import BaseHeteroModel
from .gated_gnn import HeteroGatedGNN
from .SAGE_Fraud import HeteroSAGE, SAGEConfig, run_seeds, train_model

# Registry: maps model name → class.
# Add new models here as they are implemented.
# NOTE: HeteroSAGE is not registered here -- it doesn't implement the
# BaseHeteroModel / fit_refinement constructor contract and has its own
# standalone train_model()/run_seeds() training loop.
MODELS: dict = {
    "gated_gnn": HeteroGatedGNN,
}

__all__ = [
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    "HeteroSAGE", "SAGEConfig", "run_seeds", "train_model",
]
