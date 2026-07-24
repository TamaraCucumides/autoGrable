"""
Heterogeneous GraphSAGE baseline for AutoGrable. 

Design constraints this file is built around:

  1. `num_layers=0` recovers just MLPHead, i.e. the trivial-grable baseline, sharing
     every component with the GNN except message passing. 

  2. Value nodes carry no features (build_hetero_graph writes zeros[N,1]).
     They are initialised from a per-column learnable prototype. Optional
     symmetry-breaking mechanisms (random hash, per-ID embedding) are behind
     flags and OFF by default

  3. Loss and metrics are computed on seed nodes only under mini-batching.

  4. Binary imbalanced target: single logit, BCEWithLogitsLoss(pos_weight),
     model selection on validation average precision.

Works with either:
  - a unified transductive HeteroData with train/val/test masks on "row", or
  - three per-split graphs (inductive), by passing them separately.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

TARGET = "row"


def _key(s) -> str:
    """ModuleDict keys cannot contain '.'; edge types must be flattened."""
    if isinstance(s, tuple):
        s = "__".join(s)
    return str(s).replace(".", "_dot_").replace(" ", "_")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SAGEConfig:
    hidden_dim: int = 128
    num_layers: int = 2          # 0 => no message passing (trivial grable)
    head_layers: int = 2         # >=2 keeps the no-MP baseline a real MLP
    dropout: float = 0.2
    input_dropout: float = 0.0
    conv_aggr: str = "sum"       # within an edge type; 'sum' preserves counts
    relation_aggr: str = "sum"   # across edge types at a destination node
    residual: bool = True
    norm: str = "layer"          # 'layer' | 'batch' | 'none'

    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 40           # early stopping on val AP
    grad_clip: float = 1.0

    pos_weight: Optional[float] = None   # None => neg/pos from train labels
    pos_weight_cap: float = 50.0

    # Expressivity knobs. Both exceed 1-WL. Keep False for headline numbers.
    value_random_hash: bool = False
    value_id_embedding: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #

def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class NodeEncoder(nn.Module):
    """Maps every node type into R^H before any message passing.

    row       : Linear(D, H) over standardised tabular features (or a prototype
                if the row store has no x).
    <col>     : per-column learnable prototype, broadcast to all value nodes.
                Optionally + fixed random hash and/or per-ID embedding.
    """

    def __init__(
        self,
        node_types: Iterable[str],
        row_feat_dim: int,
        num_value_nodes: Dict[str, int],
        cfg: SAGEConfig,
    ):
        super().__init__()
        H = cfg.hidden_dim
        self.cfg = cfg
        self.node_types = list(node_types)
        self.row_feat_dim = row_feat_dim

        if row_feat_dim > 0:
            self.row_proj = nn.Linear(row_feat_dim, H)
        else:
            self.row_proto = nn.Parameter(torch.zeros(1, H))

        self.val_protos = nn.ParameterDict(
            {_key(c): nn.Parameter(torch.zeros(H)) for c in num_value_nodes}
        )

        if cfg.value_random_hash:
            # Fixed, non-trainable. Breaks WL symmetry -- ablation only.
            g = torch.Generator().manual_seed(cfg.seed)
            for c, n in num_value_nodes.items():
                self.register_buffer(
                    f"hash_{_key(c)}",
                    torch.randn(n, H, generator=g) / math.sqrt(H),
                    persistent=False,
                )

        if cfg.value_id_embedding:
            self.val_emb = nn.ModuleDict(
                {_key(c): nn.Embedding(n, H) for c, n in num_value_nodes.items()}
            )

        self.norms = nn.ModuleDict(
            {_key(t): _make_norm(cfg.norm, H) for t in self.node_types}
        )
        self.drop = nn.Dropout(cfg.input_dropout)

    def forward(self, batch: HeteroData) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        dev = next(self.parameters()).device

        for t in batch.node_types:
            n = batch[t].num_nodes
            if t == TARGET:
                if self.row_feat_dim > 0:
                    h = self.row_proj(batch[t].x.to(dev).float())
                else:
                    h = self.row_proto.expand(n, -1)
            else:
                h = self.val_protos[_key(t)].unsqueeze(0).expand(n, -1)
                # n_id is present under NeighborLoader; identifies global ids.
                gid = getattr(batch[t], "n_id", None)
                if self.cfg.value_random_hash:
                    hb = getattr(self, f"hash_{_key(t)}")
                    h = h + (hb[gid] if gid is not None else hb[:n]).to(dev)
                if self.cfg.value_id_embedding:
                    idx = gid if gid is not None else torch.arange(n, device=dev)
                    h = h + self.val_emb[_key(t)](idx.to(dev))
            out[t] = self.drop(self.norms[_key(t)](h))
        return out


class MLPHead(nn.Module):
    """Non-trivial head. Carries the whole model when num_layers == 0."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, cfg: SAGEConfig):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(cfg.head_layers - 1, 0)):
            layers += [
                nn.Linear(d, hidden),
                _make_norm(cfg.norm, hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        row_feat_dim: int,
        num_value_nodes: Dict[str, int],
        cfg: SAGEConfig,
        out_dim: int = 1,
    ):
        super().__init__()
        node_types, edge_types = metadata
        H = cfg.hidden_dim
        self.cfg = cfg
        self.node_types = node_types

        self.encoder = NodeEncoder(node_types, row_feat_dim, num_value_nodes, cfg)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        et: SAGEConv((H, H), H, aggr=cfg.conv_aggr)
                        for et in edge_types
                    },
                    aggr=cfg.relation_aggr,
                )
            )
            self.layer_norms.append(
                nn.ModuleDict({_key(t): _make_norm(cfg.norm, H) for t in node_types})
            )

        self.drop = nn.Dropout(cfg.dropout)
        # Skip from the encoder output: makes the GNN a strict superset of the
        # MLP in capacity. See the note in the accompanying discussion -- this
        # is a deliberate choice and should be ablated.
        self.head = MLPHead(H, H, out_dim, cfg)

    def forward(self, batch: HeteroData) -> Tensor:
        x = self.encoder(batch)

        for conv, norms in zip(self.convs, self.layer_norms):
            x_new = conv(x, batch.edge_index_dict)
            merged: Dict[str, Tensor] = {}
            for t in x:
                h = x_new.get(t)
                if h is None:            # no incoming edges in this batch
                    merged[t] = x[t]
                    continue
                h = norms[_key(t)](h)
                h = F.relu(h)
                h = self.drop(h)
                merged[t] = x[t] + h if self.cfg.residual else h
            x = merged

        return self.head(x[TARGET])      # [num_row_nodes, out_dim]


# --------------------------------------------------------------------------- #
# Feature standardisation (fit on train only)
# --------------------------------------------------------------------------- #

def fit_row_scaler(x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    xt = x[mask] if mask is not None else x
    mu = xt.mean(0)
    sd = xt.std(0).clamp_min(1e-6)
    return mu, sd


def apply_row_scaler(data: HeteroData, mu: Tensor, sd: Tensor) -> None:
    data[TARGET].x = (data[TARGET].x - mu) / sd


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def binary_metrics(y: np.ndarray, score: np.ndarray, thr: Optional[float] = None):
    out = {
        "ap": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
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

@torch.no_grad()
def _predict(model, data, loader, mask, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if loader is None:                                   # full batch
        logits = model(data.to(device))[:, 0]
        return (
            data[TARGET].y[mask].cpu().numpy(),
            logits[mask].cpu().numpy(),
        )
    ys, ss = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch[TARGET].batch_size
        ss.append(model(batch)[:bs, 0].cpu().numpy())
        ys.append(batch[TARGET].y[:bs].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def train_model(
    data: HeteroData,
    cfg: SAGEConfig,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    train_loader=None,
    val_loader=None,
    test_loader=None,
):
    """Full-batch when loaders are None. Prefer full-batch for count/duplicate
    tasks: sampled fan-in corrupts sum-aggregated degree information."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    row_x = data[TARGET].get("x")
    row_dim = int(row_x.shape[1]) if row_x is not None else 0
    n_values = {t: int(data[t].num_nodes) for t in data.node_types if t != TARGET}

    model = HeteroSAGE(data.metadata(), row_dim, n_values, cfg, out_dim=1).to(device)

    y = data[TARGET].y.float()
    if cfg.pos_weight is None:
        pos = float(y[train_mask].sum())
        neg = float(train_mask.sum()) - pos
        pw = min(neg / max(pos, 1.0), cfg.pos_weight_cap)
    else:
        pw = cfg.pos_weight
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(cfg.patience // 4, 3)
    )

    best_ap, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        if train_loader is None:
            d = data.to(device)
            opt.zero_grad()
            logits = model(d)[:, 0]
            loss = loss_fn(logits[train_mask], y.to(device)[train_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss = float(loss.item())
        else:
            tot, nb = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                bs = batch[TARGET].batch_size
                opt.zero_grad()
                logits = model(batch)[:bs, 0]                   # seed nodes only
                loss = loss_fn(logits, batch[TARGET].y[:bs].float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tot += float(loss.item()); nb += 1
            tr_loss = tot / max(nb, 1)

        yv, sv = _predict(model, data, val_loader, val_mask, device)
        m = binary_metrics(yv, sv)
        sched.step(m["ap"])
        history.append({"epoch": epoch, "train_loss": tr_loss, **m})

        if m["ap"] > best_ap + 1e-5:
            best_ap, bad = m["ap"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)

    yv, sv = _predict(model, data, val_loader, val_mask, device)
    thr = best_f1_threshold(yv, sv)
    yt, st = _predict(model, data, test_loader, test_mask, device)

    return {
        "model": model,
        "history": history,
        "val": binary_metrics(yv, sv, thr),
        "test": binary_metrics(yt, st, thr),
        "threshold": thr,
        "pos_weight": pw,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def run_seeds(data, cfg: SAGEConfig, masks, seeds=(0, 1, 2, 3, 4), **loaders):
    """Report mean +- std over seeds. Single-seed GNN numbers are not evidence."""
    rows = []
    for s in seeds:
        c = copy.replace(cfg, seed=s) if hasattr(copy, "replace") else _with_seed(cfg, s)
        r = train_model(data, c, *masks, **loaders)
        rows.append({"seed": s, **{f"val_{k}": v for k, v in r["val"].items()},
                     **{f"test_{k}": v for k, v in r["test"].items()},
                     "num_params": r["num_params"]})
    return rows


def _with_seed(cfg: SAGEConfig, s: int) -> SAGEConfig:
    c = copy.deepcopy(cfg)
    c.seed = s
    return c