from .types import AutoGrableConfig, AutoGrableResult, RefinementConfig
from .core import fit_autograble
from .graph import build_hetero_graph
from .refine import RefinementResult, fit_refinement, fit_gated_gnn, gate_summary
from .preprocess import make_tabular_features
from .models import (
    BaseHeteroModel,
    HeteroGatedGNN,
    HeteroSAGE,
    MODELS,
    SAGEConfig,
    _with_seed,
    apply_row_scaler,
    fit_row_scaler,
    run_seeds,
    train_model,
)
from .evaluate_graph_incidence import compute_J_incidence_from_df

__all__ = [
    # Core: autoGrable structural partition selection
    "AutoGrableConfig", "AutoGrableResult", "fit_autograble",
    # Graph builder
    "build_hetero_graph",
    # Refinement (optional): parametric GNN trained on top of the selected structure
    "RefinementConfig", "RefinementResult", "fit_refinement", "fit_gated_gnn", "gate_summary",
    # Models
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    # Standalone SAGE baseline (own train/eval loop, not routed through fit_refinement)
    "HeteroSAGE", "SAGEConfig", "train_model", "run_seeds",
    "fit_row_scaler", "apply_row_scaler", "_with_seed",
    # Preprocessing
    "make_tabular_features",
    # Evaluate graph (via J)
    "compute_J_incidence_from_df"
]
