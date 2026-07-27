from .base import BaseHeteroModel
from .gated_gnn import HeteroGatedGNN
from .SAGE_Fraud import (
    HeteroSAGE,
    SAGEConfig,
    _with_seed,
    apply_row_scaler,
    fit_row_scaler,
    run_seeds,
    train_model,
)
from .SAGE_TabArena import (
    HeteroSAGE as HeteroSAGETabArena,
    SAGEConfig as SAGETabArenaConfig,
    _with_seed as _with_seed_tabarena,
    apply_row_scaler as apply_row_scaler_tabarena,
    fit_row_scaler as fit_row_scaler_tabarena,
    run_seeds as run_seeds_tabarena,
    train_model as train_model_tabarena,
)
from .SAGE_Relbench import (
    HeteroSAGE as HeteroSAGERelbench,
    SAGEConfig as SAGERelbenchConfig,
    _with_seed as _with_seed_relbench,
    run_seeds as run_seeds_relbench,
    train_model as train_model_relbench,
)

# Registry: maps model name → class.
# Add new models here as they are implemented.
# NOTE: HeteroSAGE / HeteroSAGETabArena / HeteroSAGERelbench are not
# registered here -- they don't implement the BaseHeteroModel /
# fit_refinement constructor contract and have their own standalone
# train_model()/run_seeds() training loops.
MODELS: dict = {
    "gated_gnn": HeteroGatedGNN,
}

__all__ = [
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    "HeteroSAGE", "SAGEConfig", "run_seeds", "train_model",
    "fit_row_scaler", "apply_row_scaler", "_with_seed",
    "HeteroSAGETabArena", "SAGETabArenaConfig", "run_seeds_tabarena", "train_model_tabarena",
    "fit_row_scaler_tabarena", "apply_row_scaler_tabarena", "_with_seed_tabarena",
    "HeteroSAGERelbench", "SAGERelbenchConfig", "run_seeds_relbench", "train_model_relbench",
    "_with_seed_relbench",
]
