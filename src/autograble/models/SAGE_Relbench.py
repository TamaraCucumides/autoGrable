"""
Heterogeneous GraphSAGE baseline for AutoGrable -- relbench variant.

Unlike SAGE_Fraud.py / SAGE_TabArena.py, this file does not build its own
node encoder / message-passing stack on top of AutoGrable's `HeteroData`
convention (a single "row" entity table plus zero-featured "value" nodes).
Instead it wraps relbench's own reference GNN baseline --
`HeteroEncoder` + `HeteroTemporalEncoder` + `HeteroGraphSAGE` + an `MLP`
head -- for relbench's native heterogeneous, temporal, multi-table graphs
(arbitrary entity tables, `col_stats_dict` from `torch_frame`, seed-time
temporal sampling via `NeighborLoader`).

Design notes:

  1. Training is always mini-batched over relbench's `NeighborLoader`s
     (`loader_dict["train"/"val"/"test"]`) -- relbench graphs are sampled
     temporally per seed time, so there is no full-batch mode here (compare
     SAGE_Fraud, which prefers full-batch for count/duplicate tasks).

  2. This wrapper targets relbench's binary entity-classification tasks:
     a single logit per seed node, `BCEWithLogitsLoss`, model selection on
     a validation metric (AUROC by default; see `SAGEConfig.selection_metric`).
     Regression / multiclass relbench tasks need a different loss and head
     and are out of scope here.

  3. relbench's leaderboard protocol holds out test labels, so `train_model`
     cannot compute test metrics locally -- it returns raw `test_pred`
     scores to be scored externally via `task.evaluate(...)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SAGEConfig:
    channels: int = 128
    out_channels: int = 1
    num_layers: int = 2
    aggr: str = "sum"            # message aggregation within HeteroGraphSAGE
    norm: str = "batch_norm"     # head MLP norm: 'batch_norm' | 'layer_norm' | None
    shallow_list: List[NodeType] = field(default_factory=list)  # node types w/ extra per-ID embedding
    id_awareness: bool = False   # set True only for ranking/link-prediction readouts

    lr: float = 5e-3
    weight_decay: float = 0.0
    epochs: int = 20
    patience: int = 5            # early stopping on val selection_metric
    grad_clip: Optional[float] = None  # None => no clipping, matching relbench's reference loop

    selection_metric: str = "auroc"  # "auroc" | "ap"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    verbose: bool = False        # print per-epoch train/val metrics
    log_every: int = 1           # print every N epochs when verbose


# --------------------------------------------------------------------------- #
# Model (relbench's reference heterogeneous GraphSAGE)
# --------------------------------------------------------------------------- #

class HeteroSAGE(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def binary_metrics(y: np.ndarray, score: np.ndarray, thr: Optional[float] = None):
    out = {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, score)),
    }
    if thr is not None:
        out["f1"] = float(f1_score(y, (score >= thr).astype(int), zero_division=0))
    return out


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    qs = np.quantile(score, np.linspace(0.50, 0.999, 200))
    f1s = [f1_score(y, (score >= t).astype(int), zero_division=0) for t in qs]
    return float(qs[int(np.argmax(f1s))])


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_epoch(
    model: HeteroSAGE,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    entity_table: NodeType,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def predict(
    model: HeteroSAGE,
    loader: NeighborLoader,
    device: torch.device,
    entity_table: NodeType,
) -> np.ndarray:
    """No labels assumed available -- use for the test split, whose labels
    relbench holds out under its leaderboard protocol."""
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


@torch.no_grad()
def _predict_with_labels(
    model: HeteroSAGE,
    loader: NeighborLoader,
    device: torch.device,
    entity_table: NodeType,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like `predict`, but also gathers ground-truth labels. Only valid on
    splits whose loader batches carry `y` (train/val)."""
    model.eval()

    y_list, pred_list = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        y_list.append(batch[entity_table].y.detach().cpu())
    return torch.cat(y_list, dim=0).numpy(), torch.cat(pred_list, dim=0).numpy()


def train_model(
    data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    loader_dict: Dict[str, NeighborLoader],
    entity_table: NodeType,
    cfg: SAGEConfig,
):
    """`loader_dict` must have "train"/"val"/"test" `NeighborLoader` keys, as
    produced by relbench's `get_node_train_table_input` + `NeighborLoader`
    pattern for `entity_table`."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    model = HeteroSAGE(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=cfg.num_layers,
        channels=cfg.channels,
        out_channels=cfg.out_channels,
        aggr=cfg.aggr,
        norm=cfg.norm,
        shallow_list=cfg.shallow_list,
        id_awareness=cfg.id_awareness,
    ).to(device)

    loss_fn = BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(cfg.patience // 4, 1)
    )

    if cfg.selection_metric not in ("ap", "auroc"):
        raise ValueError(f"selection_metric must be 'ap' or 'auroc', got {cfg.selection_metric!r}")

    best_metric, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(cfg.epochs):
        tr_loss = train_epoch(
            model, loader_dict["train"], opt, loss_fn, device, entity_table, cfg.grad_clip
        )

        yv, sv = _predict_with_labels(model, loader_dict["val"], device, entity_table)
        m = binary_metrics(yv, sv)
        sched.step(m[cfg.selection_metric])
        history.append({"epoch": epoch, "train_loss": tr_loss, **m})

        improved = m[cfg.selection_metric] > best_metric + 1e-5
        if improved:
            best_metric, bad = m[cfg.selection_metric], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1

        if cfg.verbose and (epoch % cfg.log_every == 0 or improved or bad >= cfg.patience):
            marker = "*" if improved else ""
            print(
                f"epoch {epoch:4d} | train_loss {tr_loss:.4f} | "
                f"val_auroc {m['auroc']:.4f} | val_ap {m['ap']:.4f} | "
                f"best_{cfg.selection_metric} {best_metric:.4f} | bad {bad}/{cfg.patience} {marker}"
            )

        if bad >= cfg.patience:
            if cfg.verbose:
                print(f"early stopping at epoch {epoch} (no val_{cfg.selection_metric} improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best_state)

    yv, sv = _predict_with_labels(model, loader_dict["val"], device, entity_table)
    thr = best_f1_threshold(yv, sv)
    test_pred = predict(model, loader_dict["test"], device, entity_table)

    return {
        "model": model,
        "history": history,
        "val": binary_metrics(yv, sv, thr),
        "threshold": thr,
        # Raw scores -- relbench holds out test labels, so score these via
        # `task.evaluate(test_pred, ...)` rather than a local metric here.
        "test_pred": test_pred,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def run_seeds(
    data,
    col_stats_dict,
    loader_dict,
    entity_table: NodeType,
    cfg: SAGEConfig,
    seeds=(0, 1, 2, 3, 4),
):
    """Report mean +- std over seeds. Single-seed GNN numbers are not evidence."""
    rows = []
    for s in seeds:
        c = _with_seed(cfg, s)
        r = train_model(data, col_stats_dict, loader_dict, entity_table, c)
        rows.append({"seed": s, **{f"val_{k}": v for k, v in r["val"].items()},
                     "num_params": r["num_params"]})
    return rows


def _with_seed(cfg: SAGEConfig, s: int) -> SAGEConfig:
    c = copy.deepcopy(cfg)
    c.seed = s
    return c
