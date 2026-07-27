"""
Heterogeneous GraphSAGE baseline for AutoGrable -- relbench-style variant.

Same architecture as relbench's reference GNN baseline (node encoder +
temporal encoder + heterogeneous GraphSAGE + MLP head, with optional
shallow per-ID embeddings and an optional ID-awareness readout for
ranking/link-prediction tasks), but reimplemented on top of plain
`torch_geometric` building blocks -- same dependency footprint as
SAGE_Fraud.py / SAGE_TabArena.py (no `relbench` or `torch_frame` packages).

This is for a `HeteroData` that does *not* necessarily come from relbench's
own dataset/loader pipeline:

  1. Any node type may carry real features under `data[node_type].x`; those
     without one get a per-type learnable prototype (as in SAGE_Fraud's
     value nodes), broadcast to every node of that type.

  2. Temporal encoding is fully optional. A node type only participates if
     it has a `time` attribute; the `entity_table` batch needs a
     `seed_time` attribute for relative-time deltas to be computed against.
     If nothing in the graph carries time info, the temporal encoder is a
     no-op.

  3. `entity_table` is passed explicitly (not a fixed "row" constant) --
     the target/seed node type is whatever the caller says it is.

  4. Works with either full-batch (loader=None) or mini-batched
     (`NeighborLoader`, using the standard PyG `batch_size` / `n_id`
     attributes) training, same as SAGE_Fraud.

  5. Binary target: single logit, `BCEWithLogitsLoss(pos_weight)`, model
     selection on a validation metric (AP by default; see
     `SAGEConfig.selection_metric`).
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
from torch_geometric.typing import NodeType


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
    num_layers: int = 2
    head_layers: int = 1          # relbench's reference head is a single Linear
    dropout: float = 0.2
    input_dropout: float = 0.0
    conv_aggr: str = "sum"        # within an edge type
    relation_aggr: str = "sum"    # across edge types at a destination node
    residual: bool = True
    norm: str = "layer"           # 'layer' | 'batch' | 'none'

    # List of node types to add a shallow per-ID embedding to (relbench's
    # `shallow_list`); requires `n_id` on those node types' batches.
    shallow_list: List[NodeType] = field(default_factory=list)
    # ID-awareness embedding, added to the entity_table seed nodes; only
    # meaningful together with `forward_dst_readout` (ranking/link tasks).
    id_awareness: bool = False

    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 40
    grad_clip: float = 1.0

    pos_weight: Optional[float] = None   # None => neg/pos from train labels
    pos_weight_cap: float = 50.0

    selection_metric: str = "ap"  # "ap" | "auroc"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    verbose: bool = False
    log_every: int = 1


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

    Node types with a real `x` feature matrix get their own `Linear(D_t, H)`.
    Node types without one get a per-type learnable prototype, broadcast to
    every node of that type (mirrors SAGE_Fraud's zero-featured value nodes,
    generalised to arbitrary node types instead of a single "row" table).
    """

    def __init__(
        self,
        node_types: Iterable[str],
        feat_dims: Dict[str, int],
        cfg: SAGEConfig,
    ):
        super().__init__()
        H = cfg.hidden_dim
        self.cfg = cfg
        self.node_types = list(node_types)
        self.feat_dims = feat_dims

        self.projs = nn.ModuleDict()
        self.protos = nn.ParameterDict()
        for t in self.node_types:
            d = feat_dims.get(t, 0)
            if d > 0:
                self.projs[_key(t)] = nn.Linear(d, H)
            else:
                self.protos[_key(t)] = nn.Parameter(torch.zeros(H))

        self.norms = nn.ModuleDict(
            {_key(t): _make_norm(cfg.norm, H) for t in self.node_types}
        )
        self.drop = nn.Dropout(cfg.input_dropout)

    def forward(self, batch: HeteroData) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        dev = next(self.parameters()).device

        for t in batch.node_types:
            n = batch[t].num_nodes
            if _key(t) in self.projs:
                h = self.projs[_key(t)](batch[t].x.to(dev).float())
            else:
                h = self.protos[_key(t)].unsqueeze(0).expand(n, -1)
            out[t] = self.drop(self.norms[_key(t)](h))
        return out


class TemporalEncoder(nn.Module):
    """Sinusoidal relative-time encoding, added to node types that carry a
    `time` attribute. Purely additive and optional: if a graph has no time
    info anywhere, this module is simply never invoked with any entries."""

    def __init__(self, node_types: Iterable[str], channels: int):
        super().__init__()
        self.channels = channels
        self.lin = nn.ModuleDict(
            {_key(t): nn.Linear(channels, channels) for t in node_types}
        )

    def reset_parameters(self):
        for lin in self.lin.values():
            lin.reset_parameters()

    def _sinusoidal(self, delta: Tensor) -> Tensor:
        dev = delta.device
        freq = torch.exp(
            torch.arange(0, self.channels, 2, device=dev).float()
            * (-math.log(10000.0) / self.channels)
        )
        args = delta.unsqueeze(-1) * freq
        enc = torch.zeros(delta.size(0), self.channels, device=dev)
        enc[:, 0::2] = torch.sin(args)
        enc[:, 1 : enc.size(1) : 2] = torch.cos(args[:, : enc[:, 1::2].size(1)])
        return enc

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[str, Tensor],
        batch_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for t, time in time_dict.items():
            if _key(t) not in self.lin:
                continue
            rel = seed_time[batch_dict[t]] - time
            out[t] = self.lin[_key(t)](self._sinusoidal(rel.float()))
        return out


class MLPHead(nn.Module):
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
        feat_dims: Dict[str, int],
        num_nodes_dict: Dict[str, int],
        temporal_node_types: Iterable[str],
        cfg: SAGEConfig,
        out_dim: int = 1,
    ):
        super().__init__()
        node_types, edge_types = metadata
        H = cfg.hidden_dim
        self.cfg = cfg
        self.node_types = node_types

        self.encoder = NodeEncoder(node_types, feat_dims, cfg)
        self.temporal_encoder = TemporalEncoder(temporal_node_types, H)

        in_degree: Dict[str, int] = {}
        for _, _, dst in edge_types:
            in_degree[dst] = in_degree.get(dst, 0) + 1

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.relation_projs = nn.ModuleList()
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
            if cfg.relation_aggr == "cat":
                self.relation_projs.append(
                    nn.ModuleDict(
                        {
                            _key(t): nn.Linear(in_degree[t] * H, H)
                            for t in node_types
                            if in_degree.get(t, 0) > 1
                        }
                    )
                )
            else:
                self.relation_projs.append(nn.ModuleDict())

        self.drop = nn.Dropout(cfg.dropout)
        self.head = MLPHead(H, H, out_dim, cfg)

        self.embedding_dict = nn.ModuleDict(
            {_key(t): nn.Embedding(num_nodes_dict[t], H) for t in cfg.shallow_list}
        )
        for emb in self.embedding_dict.values():
            nn.init.normal_(emb.weight, std=0.1)

        self.id_awareness_emb = None
        if cfg.id_awareness:
            self.id_awareness_emb = nn.Embedding(1, H)

    def _seed_size(self, batch: HeteroData, entity_table: NodeType) -> int:
        bs = getattr(batch[entity_table], "batch_size", None)
        if bs is not None:
            return bs
        seed_time = getattr(batch[entity_table], "seed_time", None)
        if seed_time is not None:
            return seed_time.size(0)
        return batch[entity_table].num_nodes

    def _encode(self, batch: HeteroData, entity_table: NodeType) -> Dict[str, Tensor]:
        x_dict = self.encoder(batch)

        seed_time = getattr(batch[entity_table], "seed_time", None)
        time_dict = getattr(batch, "time_dict", None)
        if seed_time is not None and time_dict:
            rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch.batch_dict)
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type in self.cfg.shallow_list:
            embedding = self.embedding_dict[_key(node_type)]
            gid = getattr(batch[node_type], "n_id", None)
            idx = gid if gid is not None else torch.arange(
                batch[node_type].num_nodes, device=x_dict[node_type].device
            )
            x_dict[node_type] = x_dict[node_type] + embedding(idx.to(x_dict[node_type].device))

        return x_dict

    def _gnn(self, x_dict: Dict[str, Tensor], batch: HeteroData) -> Dict[str, Tensor]:
        x = x_dict
        for conv, norms, projs in zip(self.convs, self.layer_norms, self.relation_projs):
            x_new = conv(x, batch.edge_index_dict)
            merged: Dict[str, Tensor] = {}
            for t in x:
                h = x_new.get(t)
                if h is None:
                    merged[t] = x[t]
                    continue
                if _key(t) in projs:
                    h = projs[_key(t)](h)
                h = norms[_key(t)](h)
                h = F.relu(h)
                h = self.drop(h)
                merged[t] = x[t] + h if self.cfg.residual else h
            x = merged
        return x

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        x_dict = self._encode(batch, entity_table)
        x_dict = self._gnn(x_dict, batch)
        n = self._seed_size(batch, entity_table)
        return self.head(x_dict[entity_table][:n])

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
        x_dict = self._encode(batch, entity_table)
        n = self._seed_size(batch, entity_table)
        x_dict[entity_table] = x_dict[entity_table].clone()
        x_dict[entity_table][:n] = x_dict[entity_table][:n] + self.id_awareness_emb.weight
        x_dict = self._gnn(x_dict, batch)
        return self.head(x_dict[dst_table])


# --------------------------------------------------------------------------- #
# Feature standardisation (fit on train only)
# --------------------------------------------------------------------------- #

def fit_row_scaler(x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    xt = x[mask] if mask is not None else x
    mu = xt.mean(0)
    sd = xt.std(0).clamp_min(1e-6)
    return mu, sd


def apply_row_scaler(data: HeteroData, node_type: str, mu: Tensor, sd: Tensor) -> None:
    data[node_type].x = (data[node_type].x - mu) / sd


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
def _predict(model, data, loader, mask, entity_table, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if loader is None:
        d = data.to(device)
        logits = model(d, entity_table)[:, 0]
        return (
            data[entity_table].y[mask].cpu().numpy(),
            logits[mask].cpu().numpy(),
        )
    ys, ss = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch[entity_table].batch_size
        ss.append(model(batch, entity_table)[:bs, 0].cpu().numpy())
        ys.append(batch[entity_table].y[:bs].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def train_model(
    data: HeteroData,
    entity_table: NodeType,
    cfg: SAGEConfig,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    train_loader=None,
    val_loader=None,
    test_loader=None,
):
    """Full-batch when loaders are None, mini-batched (`NeighborLoader`)
    otherwise -- see SAGE_Fraud.train_model for the same convention."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    feat_dims = {
        t: int(data[t].x.shape[1]) for t in data.node_types if data[t].get("x") is not None
    }
    temporal_node_types = [t for t in data.node_types if "time" in data[t]]

    model = HeteroSAGE(
        data.metadata(), feat_dims, data.num_nodes_dict, temporal_node_types, cfg, out_dim=1
    ).to(device)

    y = data[entity_table].y.float()
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

    if cfg.selection_metric not in ("ap", "auroc"):
        raise ValueError(f"selection_metric must be 'ap' or 'auroc', got {cfg.selection_metric!r}")

    best_metric, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        if train_loader is None:
            d = data.to(device)
            opt.zero_grad()
            logits = model(d, entity_table)[:, 0]
            loss = loss_fn(logits[train_mask], y.to(device)[train_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss = float(loss.item())
        else:
            tot, nb = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                bs = batch[entity_table].batch_size
                opt.zero_grad()
                logits = model(batch, entity_table)[:bs, 0]
                loss = loss_fn(logits, batch[entity_table].y[:bs].float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tot += float(loss.item()); nb += 1
            tr_loss = tot / max(nb, 1)

        yv, sv = _predict(model, data, val_loader, val_mask, entity_table, device)
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
                f"val_ap {m['ap']:.4f} | val_auroc {m['auroc']:.4f} | "
                f"best_{cfg.selection_metric} {best_metric:.4f} | bad {bad}/{cfg.patience} {marker}"
            )

        if bad >= cfg.patience:
            if cfg.verbose:
                print(f"early stopping at epoch {epoch} (no val_{cfg.selection_metric} improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best_state)

    yv, sv = _predict(model, data, val_loader, val_mask, entity_table, device)
    thr = best_f1_threshold(yv, sv)
    yt, st = _predict(model, data, test_loader, test_mask, entity_table, device)

    return {
        "model": model,
        "history": history,
        "val": binary_metrics(yv, sv, thr),
        "test": binary_metrics(yt, st, thr),
        "threshold": thr,
        "pos_weight": pw,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def run_seeds(data, entity_table: NodeType, cfg: SAGEConfig, masks, seeds=(0, 1, 2, 3, 4), **loaders):
    """Report mean +- std over seeds. Single-seed GNN numbers are not evidence."""
    rows = []
    for s in seeds:
        c = _with_seed(cfg, s)
        r = train_model(data, entity_table, c, *masks, **loaders)
        rows.append({"seed": s, **{f"val_{k}": v for k, v in r["val"].items()},
                     **{f"test_{k}": v for k, v in r["test"].items()},
                     "num_params": r["num_params"]})
    return rows


def _with_seed(cfg: SAGEConfig, s: int) -> SAGEConfig:
    c = copy.deepcopy(cfg)
    c.seed = s
    return c
