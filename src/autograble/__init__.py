from .types import AutoGrableConfig, AutoGrableResult, RefinementConfig
from .core import fit_autograble
from .graph import build_hetero_graph, build_hetero_graph_from_joined_table
from .refine import RefinementResult, fit_refinement, fit_gated_gnn, gate_summary
from .preprocess import make_tabular_features
from .models import (
    BaseHeteroModel,
    HeteroGatedGNN,
    HeteroSAGE,
    HeteroSAGETabArena,
    MODELS,
    SAGEConfig,
    SAGETabArenaConfig,
    _with_seed,
    _with_seed_tabarena,
    apply_row_scaler,
    apply_row_scaler_tabarena,
    fit_row_scaler,
    fit_row_scaler_tabarena,
    run_seeds,
    run_seeds_tabarena,
    train_model,
    train_model_tabarena,
)
from .evaluate_graph_incidence import compute_J_incidence_from_df

__all__ = [
    # Core: autoGrable structural partition selection
    "AutoGrableConfig", "AutoGrableResult", "fit_autograble",
    # Graph builder
    "build_hetero_graph", "build_hetero_graph_from_joined_table",
    # Refinement (optional): parametric GNN trained on top of the selected structure
    "RefinementConfig", "RefinementResult", "fit_refinement", "fit_gated_gnn", "gate_summary",
    # Models
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    # Standalone SAGE baseline (own train/eval loop, not routed through fit_refinement)
    "HeteroSAGE", "SAGEConfig", "train_model", "run_seeds",
    "fit_row_scaler", "apply_row_scaler", "_with_seed",
    # Standalone SAGE baseline for TabArena
    "HeteroSAGETabArena", "SAGETabArenaConfig", "train_model_tabarena", "run_seeds_tabarena",
    "fit_row_scaler_tabarena", "apply_row_scaler_tabarena", "_with_seed_tabarena",
    # Preprocessing
    "make_tabular_features",
    # Evaluate graph (via J)
    "compute_J_incidence_from_df"
]
