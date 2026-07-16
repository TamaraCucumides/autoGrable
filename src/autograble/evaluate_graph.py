"""
training-free alignment score J for a constructed graph.

J(pi) = Risk_val(h_pi) + lambda * Omega(train, pi)

where
    pi          = 1-WL colour partition of the TARGET (row) nodes of the graph,
    h_pi        = block predictor: per-block empirical class distribution on train,
                  falling back to the global prior for blocks unseen at train time,
    Risk_val    = loss of h_pi on the held-out targets (log-loss or 0-1),
    Omega       = (1/n_tr) * sum_{B : n_B>0} sqrt(n_B), the occupancy penalty.

Reading: J estimates the lowest error a 1-WL-bounded learner can achieve on this
graph (Risk_val ~ approximation gap) plus its estimation fragility (Omega). Low J
=> the graph admits a good 1-WL predictor for this task; high J => none exists.

    * init_color_fn : feature-seeded (predicts a featureful GNN) vs structure-only
                      (degree/type-seeded, isolates the construction's contribution).
    * n_rounds      : None -> run to the WL fixpoint (a ceiling); k -> match a
                      k-layer GNN's depth (a predictor for that GNN).
    * train_idx/val_idx : the split is what prevents the degenerate "purity" reading
                      (a shattered partition scores badly on val, not on train).
                      Pass a leakage-safe split (temporal/grouped) explicitly.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Optional, Sequence

NodeId = Hashable
Color = int
Label = Hashable

EPS = 1e-12


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class JResult:
    J: float
    risk_val: float          # the "alignment error": 1-WL-achievable risk estimate
    omega: float             # where on the refinement lattice pi sits
    lam: float
    num_blocks: int          # |pi| on the targets
    block_of: dict[int, Color]  # target position -> block id (the partition itself)


# --------------------------------------------------------------------------- #
# 1-WL colour refinement  (mechanical, fully specified — leave as is)
# --------------------------------------------------------------------------- #
def _stable_hash(obj) -> Color:
    # deterministic across processes (unlike builtin hash, which is salted)
    return int.from_bytes(hashlib.blake2b(repr(obj).encode(), digest_size=8).digest(), "big")


def wl_refinement(
    nodes: Sequence[NodeId],
    neighbors_fn: Callable[[NodeId], Iterable[tuple[NodeId, Hashable]]],
    init_color_fn: Callable[[NodeId], Hashable],
    n_rounds: Optional[int] = None,
) -> dict[NodeId, Color]:
    """1-WL / colour refinement on the WHOLE graph.

    neighbors_fn(v) yields (neighbour, edge_type) pairs; edge_type may be None
    for homogeneous graphs. Stops at the fixpoint (partition stops refining) when
    n_rounds is None, else after exactly n_rounds rounds.
    """
    nodes = list(nodes)
    color: dict[NodeId, Color] = {v: _stable_hash(init_color_fn(v)) for v in nodes}
    prev_n = len(set(color.values()))
    r = 0
    while True:
        new_color: dict[NodeId, Color] = {}
        for v in nodes:
            multiset = tuple(sorted((et, color[u]) for u, et in neighbors_fn(v)))
            new_color[v] = _stable_hash((color[v], multiset))
        color = new_color
        r += 1
        n = len(set(color.values()))
        if n_rounds is not None:
            if r >= n_rounds:
                break
        else:
            if n == prev_n:      # refinement stabilised -> fixpoint
                break
            prev_n = n
    return color


# --------------------------------------------------------------------------- #
# block predictor, validation risk, occupancy  (mechanical — fully specified)
# --------------------------------------------------------------------------- #
def _normalize(counts: dict[Label, float], classes: Sequence[Label]) -> dict[Label, float]:
    total = sum(counts.values())
    if total <= 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: counts.get(c, 0.0) / total for c in classes}


def fit_block_predictor(
    block_of: dict[int, Color],
    labels: Sequence[Label],
    train_idx: Sequence[int],
    classes: Sequence[Label],
) -> tuple[dict[Color, dict[Label, float]], dict[Label, float]]:
    block_counts: dict[Color, dict[Label, float]] = defaultdict(lambda: defaultdict(float))
    prior_counts: dict[Label, float] = defaultdict(float)
    for i in train_idx:
        block_counts[block_of[i]][labels[i]] += 1.0
        prior_counts[labels[i]] += 1.0
    block_dist = {b: _normalize(c, classes) for b, c in block_counts.items()}
    prior = _normalize(prior_counts, classes)
    return block_dist, prior


def val_risk(
    block_of: dict[int, Color],
    block_dist: dict[Color, dict[Label, float]],
    prior: dict[Label, float],
    labels: Sequence[Label],
    val_idx: Sequence[int],
    loss: str = "log",
) -> float:
    if not val_idx:
        return float("nan")
    total = 0.0
    for i in val_idx:
        dist = block_dist.get(block_of[i], prior)   # unseen block -> prior
        y = labels[i]
        if loss == "log":
            total += -math.log(max(dist.get(y, 0.0), EPS))
        elif loss == "01":
            pred = max(dist, key=dist.get)
            total += 0.0 if pred == y else 1.0
        else:
            raise ValueError(f"unknown loss {loss!r}")
    return total / len(val_idx)


def occupancy(block_of: dict[int, Color], train_idx: Sequence[int]) -> float:
    ntr = len(train_idx)
    if ntr == 0:
        return float("nan")
    n_B = Counter(block_of[i] for i in train_idx)
    return sum(math.sqrt(c) for c in n_B.values()) / ntr


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def compute_J(
    all_nodes: Sequence[NodeId],
    target_nodes: Sequence[NodeId],
    labels: Sequence[Label],
    *,
    neighbors_fn: Callable[[NodeId], Iterable[tuple[NodeId, Hashable]]],
    init_color_fn: Callable[[NodeId], Hashable],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    lam: float = 1.0,
    n_rounds: Optional[int] = None,
    loss: str = "log",
) -> JResult:
    """Compute J on an arbitrary graph.

    target_nodes and labels are parallel sequences (labels[i] is the label of
    target_nodes[i]); train_idx / val_idx are index positions into them.
    all_nodes is every node in the graph (WL runs on the whole graph, then the
    colouring is restricted to target_nodes).
    """
    colors = wl_refinement(all_nodes, neighbors_fn, init_color_fn, n_rounds)
    block_of = {i: colors[target_nodes[i]] for i in range(len(target_nodes))}
    classes = sorted(set(labels))

    block_dist, prior = fit_block_predictor(block_of, labels, train_idx, classes)
    rv = val_risk(block_of, block_dist, prior, labels, val_idx, loss)
    om = occupancy(block_of, train_idx)

    return JResult(
        J=rv + lam * om,
        risk_val=rv,
        omega=om,
        lam=lam,
        num_blocks=len(set(block_of.values())),
        block_of=block_of,
    )


# --------------------------------------------------------------------------- #
# adapters & knobs — TODO: wire these to your data / graph library
# --------------------------------------------------------------------------- #
# --- graph I/O -------------------------------------------------------------- #
def networkx_neighbors(G, edge_type_attr: Optional[str] = None):
    """TODO: adapter for networkx. Returns a neighbors_fn.
    edge_type_attr names the edge attribute holding the (typed) relation, or None.
    For incidence grables the edge type = the column; keep it, it matters for WL.
    """
    def fn(v):
        for u in G.neighbors(v):
            et = G[v][u].get(edge_type_attr) if edge_type_attr else None
            yield (u, et)
    return fn


def pyg_neighbors(data):
    """TODO: adapter for a torch_geometric Data/HeteroData object.
    Build an adjacency once (edge_index -> per-node list of (neighbour, edge_type))
    and close over it; don't rescan edge_index per call.
    """
    raise NotImplementedError


# --- initial colouring: the feature-vs-structure knob ----------------------- #
def structure_init(degree_fn: Callable[[NodeId], int], type_fn: Optional[Callable] = None):
    """Structure-only seed (degree / node-type). Isolates what the CONSTRUCTION
    adds, independent of features. Use for the 'structural contribution' reading.
    """
    def fn(v):
        return ("struct", degree_fn(v), None if type_fn is None else type_fn(v))
    return fn


def feature_init(feature_fn: Callable[[NodeId], Hashable]):
    """Feature/type seed. Use when J should PREDICT a featureful GNN — i.e. WL
    starts from the same information the GNN sees at layer 0.
    TODO: feature_fn must return a hashable, discretised signature (bucket/round
    continuous features, else every node is its own colour and WL over-refines).
    """
    def fn(v):
        return ("feat", feature_fn(v))
    return fn


# --- split ------------------------------------------------------------------ #
def random_split(n: int, val_frac: float = 0.3, seed: int = 0):
    """TODO: placeholder ONLY. Replace with the leakage-safe split for your data
    (temporal cutoff, or grouped by entity/primary key). A random split over
    target rows leaks in temporal/relational settings — see the draft's split logic.
    """
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(round((1 - val_frac) * n))
    return idx[:cut], idx[cut:]


# --------------------------------------------------------------------------- #
# usage sketch (delete once wired up)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # graph          : your constructed grable (nx / pyg / custom)
    # target_nodes   : the row nodes you predict on
    # labels         : parallel to target_nodes
    #
    # neigh = networkx_neighbors(G, edge_type_attr="column")
    # init  = structure_init(degree_fn=G.degree)        # or feature_init(...)
    # tr, va = random_split(len(target_nodes))          # or a leakage-safe split
    # res = compute_J(list(G.nodes), target_nodes, labels,
    #                 neighbors_fn=neigh, init_color_fn=init,
    #                 train_idx=tr, val_idx=va, lam=1.0, n_rounds=None)
    # print(res.J, res.risk_val, res.omega, res.num_blocks)
    pass