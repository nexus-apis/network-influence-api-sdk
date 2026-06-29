# core.py — Network Influence API: Causal PageRank Core Module

```python
"""
core.py
=======
Network Influence API — Core Module
Implements CausalEdgeWeighter, BootstrapCIEngine, CausalPageRankSolver,
CounterfactualScorer, DAGAuditExporter, and SDKCore.

Dependencies:
    pip install networkx pgmpy statsmodels causal-learn numpy scipy
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy import stats

# ── statsmodels (Granger) ───────────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not found; Granger causality disabled.")

# ── causal-learn (PC Algorithm) ────────────────────────────────────────────
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    HAS_CAUSALLEARN = True
except ImportError:  # pragma: no cover
    HAS_CAUSALLEARN = False
    warnings.warn("causal-learn not found; PC algorithm disabled.")

# ── pgmpy (do-calculus utilities) ──────────────────────────────────────────
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.inference import CausalInference
    HAS_PGMPY = True
except ImportError:  # pragma: no cover
    HAS_PGMPY = False
    warnings.warn("pgmpy not found; do-calculus CounterfactualScorer disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("causal_pagerank")

__version__ = "0.1.0"
__all__ = [
    "CausalEdgeWeighter",
    "BootstrapCIEngine",
    "CausalPageRankSolver",
    "CounterfactualScorer",
    "DAGAuditExporter",
    "SDKCore",
    "CausalEdge",
    "CausalDAG",
    "CausalPageRankResult",
    "CounterfactualResult",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalEdge:
    """Directed causal edge with statistical metadata."""

    source: str
    target: str
    weight: float                        # normalised causal effect size
    p_value: float                       # statistical significance
    effect_size: float                   # raw effect size (e.g. F-stat, coeff)
    method: str                          # "granger" | "pc" | "manual"
    ci_lower: float = 0.0               # bootstrap 2.5-percentile
    ci_upper: float = 0.0               # bootstrap 97.5-percentile
    lag: Optional[int] = None           # lag used (Granger only)
    confidence: float = 0.0             # 1 - p_value clipped to [0,1]

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(1.0 - self.p_value, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalDAG:
    """Annotated directed acyclic graph produced by CausalEdgeWeighter."""

    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = "unknown"
    version: str = __version__
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    input_hash: str = ""                # SHA-256 of serialised input graph
    nodes: List[str] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)
    nx_dag: Optional[nx.DiGraph] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── convenience helpers ──────────────────────────────────────────────────

    def to_networkx(self) -> nx.DiGraph:
        if self.nx_dag is not None:
            return self.nx_dag
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for e in self.edges:
            G.add_edge(
                e.source, e.target,
                weight=e.weight,
                p_value=e.p_value,
                effect_size=e.effect_size,
                method=e.method,
                ci_lower=e.ci_lower,
                ci_upper=e.ci_upper,
                confidence=e.confidence,
            )
        self.nx_dag = G
        return G

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "algorithm": self.algorithm,
            "version": self.version,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }


@dataclass
class CausalPageRankResult:
    """Output contract for CausalPageRankSolver."""

    job_id: str
    dag_id: str
    scores: Dict[str, float]            # node → causal PageRank score
    rank: List[str]                     # nodes sorted high → low
    ci_lower: Dict[str, float]          # bootstrap lower bound
    ci_upper: Dict[str, float]          # bootstrap upper bound
    alpha: float = 0.85
    epsilon: float = 1e-6
    iterations: int = 0
    converged: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    """Output contract for CounterfactualScorer."""

    removed_node: str
    baseline_scores: Dict[str, float]
    post_intervention_scores: Dict[str, float]
    delta_scores: Dict[str, float]      # post - baseline
    removal_impact: float               # aggregate Σ|Δ| / N
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CausalEdgeWeighter
# ══════════════════════════════════════════════════════════════════════════════

class CausalEdgeWeighter:
    """
    Orchestrates causal edge inference using:
      • PC Algorithm  (causal-learn) — cross-sectional / observational data
      • Granger Causality (statsmodels) — multivariate time-series data

    Parameters
    ----------
    method : str
        "pc" | "granger" | "auto"
        "auto" selects Granger when time_series=True, else PC.
    alpha : float
        Significance threshold for causal tests.
    max_lag : int
        Maximum lag for Granger tests.
    normalize_weights : bool
        If True, edge weights sum to 1 per source node (row-stochastic).
    """

    SUPPORTED_METHODS = ("pc", "granger", "auto")

    def __init__(
        self,
        method: str = "auto",
        alpha: float = 0.05,
        max_lag: int = 3,
        normalize_weights: bool = True,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {self.SUPPORTED_METHODS}")
        self.method = method
        self.alpha = alpha
        self.max_lag = max_lag
        self.normalize_weights = normalize_weights

    # ── public entry point ───────────────────────────────────────────────────

    def fit(
        self,
        data: np.ndarray,
        node_names: List[str],
        *,
        time_series: bool = False,
        input_graph: Optional[nx.Graph] = None,
    ) -> CausalDAG:
        """
        Infer causal DAG from data.

        Parameters
        ----------
        data : np.ndarray
            Shape (T, N) — T observations, N variables (nodes).
        node_names : list[str]
            Length N list of node labels.
        time_series : bool
            Treat rows as consecutive time steps (triggers Granger).
        input_graph : nx.Graph, optional
            Skeleton to restrict search space.

        Returns
        -------
        CausalDAG
        """
        if data.shape[1] != len(node_names):
            raise ValueError(
                f"data has {data.shape[1]} columns but {len(node_names)} node names."
            )

        method = self._resolve_method(time_series)
        logger.info("CausalEdgeWeighter: method=%s nodes=%d", method, len(node_names))

        input_hash = self._hash_input(data, node_names)

        if method == "granger":
            edges = self._granger_edges(data, node_names)
        else:
            edges = self._pc_edges(data, node_names)

        if self.normalize_weights:
            edges = self._normalize(edges, node_names)

        dag = CausalDAG(
            algorithm=method,
            input_hash=input_hash,
            nodes=node_names,
            edges=edges,
            metadata={
                "alpha": self.alpha,
                "max_lag": self.max_lag if method == "granger" else None,
                "n_observations": data.shape[0],
                "n_nodes": len(node_names),
            },
        )
        dag.to_networkx()          # cache nx representation
        logger.info(
            "CausalEdgeWeighter: DAG built — %d nodes, %d edges",
            len(dag.nodes), len(dag.edges),
        )
        return dag

    # ── private helpers ──────────────────────────────────────────────────────

    def _resolve_method(self, time_series: bool) -> str:
        if self.method != "auto":
            return self.method
        return "granger" if time_series else "pc"

    @staticmethod
    def _hash_input(data: np.ndarray, node_names: List[str]) -> str:
        h = hashlib.sha256()
        h.update(data.tobytes())
        h.update(json.dumps(node_names).encode())
        return h.hexdigest()

    def _granger_edges(
        self, data: np.ndarray, node_names: List[str]
    ) -> List[CausalEdge]:
        if not HAS_STATSMODELS:
            raise RuntimeError("statsmodels required for Granger causality.")

        N = len(node_names)
        edges: List[CausalEdge] = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # test: does node_i Granger-cause node_j?
                xy = data[:, [j, i]]   # statsmodels: [effect, cause]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = grangercausalitytests(
                            xy, maxlag=self.max_lag, verbose=False
                        )
                except Exception as exc:
                    logger.debug("Granger %s→%s failed: %s", node_names[i], node_names[j], exc)
                    continue

                best_lag, best_pval, best_fstat = self._best_granger_lag(results)
                if best_pval < self.alpha:
                    edges.append(
                        CausalEdge(
                            source=node_names[i],
                            target=node_names[j],
                            weight=float(best_fstat),
                            p_value=float(best_pval),
                            effect_size=float(best_fstat),
                            method="granger",
                            lag=best_lag,
                        )
                    )
        return edges

    @staticmethod
    def _best_granger_lag(
        results: Dict,
    ) -> Tuple[int, float, float]:
        """Return (lag, p_value, F_statistic) with smallest p_value."""
        best_lag = 1
        best_p = 1.0
        best_f = 0.0
        for lag, res in results.items():
            p = res[0]["ssr_ftest"][1]
            f = res[0]["ssr_ftest"][0]
            if p < best_p:
                best_p, best_f, best_lag = p, f, lag
        return best_lag, best_p, best_f

    def _pc_edges(
        self, data: np.ndarray, node_names: List[str]
    ) -> List[CausalEdge]:
        if not HAS_CAUSALLEARN:
            raise RuntimeError("causal-learn required for PC algorithm.")

        cg = pc(data, alpha=self.alpha, indep_test=fisherz, show_progress=False)
        adj = cg.G.graph                  # numpy adjacency: 1=tail, -1=arrowhead
        N = len(node_names)
        edges: List[CausalEdge] = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # directed i→j: adj[i,j]==-1 AND adj[j,i]==1
                if adj[j, i] == -1 and adj[i, j] == 1:
                    # approximate p-value from correlation for effect size
                    corr, pval = stats.pearsonr(data[:, i], data[:, j])
                    if pval < self.alpha:
                        edges.append(
                            CausalEdge(
                                source=node_names[i],
                                target=node_names[j],
                                weight=abs(float(corr)),
                                p_value=float(pval),
                                effect_size=abs(float(corr)),
                                method="pc",
                            )
                        )
        return edges

    @staticmethod
    def _normalize(
        edges: List[CausalEdge], node_names: List[str]
    ) -> List[CausalEdge]:
        """Make weights row-stochastic (sum per source = 1)."""
        source_totals: Dict[str, float] = {}
        for e in edges:
            source_totals[e.source] = source_totals.get(e.source, 0.0) + e.weight

        for e in edges:
            total = source_totals.get(e.source, 1.0)
            e.weight = e.weight / total if total > 0 else 0.0
        return edges


# ══════════════════════════════════════════════════