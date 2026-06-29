# Network Influence API

> **Causal PageRank for production pipelines** — find nodes that *actually drive* outcomes, not just nodes that co-occur with them.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status: MVP](https://img.shields.io/badge/status-MVP-orange.svg)]()
[![WebAssembly Ready](https://img.shields.io/badge/WASM-Pyodide--ready-654FF0.svg)](https://pyodide.org)

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [Core Concepts](#2-core-concepts)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Reference](#4-module-reference)
5. [API Reference](#5-api-reference)
6. [Quickstart](#6-quickstart)
7. [SDK — Zero-Infra Mode](#7-sdk--zero-infra-mode)
8. [Output Schema](#8-output-schema)
9. [Tech Stack](#9-tech-stack)
10. [Benchmarks](#10-benchmarks)
11. [Roadmap](#11-roadmap)
12. [Contributing](#12-contributing)
13. [License](#13-license)

---

## 1. Why This Exists

Standard PageRank ranks nodes by *topological prominence* — how many edges point to a node and how prominent those neighbors are. It says nothing about whether removing a node would actually change outcomes downstream.

The market offers two separate tools:

| Tool | What it does | What it misses |
|---|---|---|
| Standard PageRank / eigenvector centrality | Fast, scalable influence ranking | Correlational — cannot separate causation from co-occurrence |
| DoWhy / CausalML | Rigorous causal inference on tabular data | No native graph propagation, not production-API ready |

**Network Influence API closes that gap.**

It runs causal inference (PC Algorithm or Granger Causality, depending on your network type) to assign *directional causal weights* to every edge, then feeds those weights into a modified PageRank solver. The result is a node ranking that reflects **intervenable impact**: if you remove or suppress node X, scores across the network will shift by a predictable, auditable delta.

Every response ships with:

- A ranked node list with bootstrap confidence intervals.
- The full **certified causal DAG** (exportable as GraphML, JSON-LD, or DOT).
- **Counterfactual scores** per node computed via do-calculus (`do(X=0) → Δ influence`).
- Complete audit metadata (algorithm, version, timestamp, SHA-256 of the input graph).

---

## 2. Core Concepts

### 2.1 Causal Edge Weighting

Before any ranking is computed, the raw graph is transformed into a **causal DAG**. Two algorithms are available:

**PC Algorithm** (via `causal-learn`) — preferred for cross-sectional graphs (social networks, knowledge graphs, supply chain snapshots). It tests conditional independence between node pairs and orients edges based on v-structures. Outputs p-values and effect sizes per edge.

**Granger Causality** (via `statsmodels`) — preferred for temporal graphs (time-series activity logs, sequential supply chain events). Node A Granger-causes node B if past values of A improve the prediction of B beyond B's own history.

Both algorithms annotate each surviving edge with:

```
weight        → normalized causal effect size  [0, 1]
p_value       → statistical significance
ci_lower      → 5th-percentile bootstrap weight
ci_upper      → 95th-percentile bootstrap weight
direction     → source → target (oriented, not undirected)
```

### 2.2 Causal PageRank

Once the causal DAG is available, the transition matrix is built from **normalized causal weights** instead of inverse node degrees. Formally, for a node `j` pointing to node `i`:

```
M[i][j] = causal_weight(j → i) / Σ_k causal_weight(j → k)
```

PageRank then iterates:

```
R(t+1) = α · M · R(t) + (1 − α) · (1/N)
```

where `α` (damping) and `ε` (convergence threshold) are configurable per request.

### 2.3 Counterfactual Scoring

For each node in the top-K ranking, the API applies `do(X = 0)` on the stored DAG using do-calculus, removes all outgoing causal influence from X, re-runs Causal PageRank, and reports:

```
removal_impact  → % change in total network influence mass
delta_scores    → per-node score delta for top-K neighbors
```

This answers the question: *"If we deactivate this node, what breaks and by how much?"*

---

## 3. Architecture Overview

The pipeline is split into two **decoupled phases** with explicit data contracts. Each phase can be audited, cached, and reused independently.

```
INPUT GRAPH
(adjacency list / edge-list / GraphML)
        │
        ▼
┌───────────────────────────────────────────────┐
│  PHASE 1 — Causal Edge Weighting              │
│                                               │
│  CausalEdgeWeighter                           │
│    ├── PC Algorithm  (causal-learn)           │
│    └── Granger Causality (statsmodels)        │
│                                               │
│  BootstrapCIEngine  (N=1000 resamples)        │
│                                               │
│  Output: Certified Causal DAG                 │
│    → stored in Neo4j                          │
│    → CI distributions in PostgreSQL JSONB     │
└───────────────────────────────────────────────┘
        │
        │  Causal DAG (cached, reusable)
        ▼
┌───────────────────────────────────────────────┐
│  PHASE 2 — Causal PageRank Propagation        │
│                                               │
│  CausalPageRankSolver                         │
│    └── Transition matrix from causal weights  │
│                                               │
│  CounterfactualScorer                         │
│    └── do-calculus on Neo4j DAG               │
│                                               │
│  DAGAuditExporter                             │
│    └── GraphML / JSON-LD / DOT + SHA-256      │
└───────────────────────────────────────────────┘
        │
        ▼
   JSON RESPONSE
   + DAG export
   + webhook / polling
```

**Why this separation matters:**

- The causal DAG is expensive to compute and can be **reused across multiple analyses** without re-running inference.
- Phase 2 can be packaged as a **pure Python / WASM module** (no Neo4j, no Celery) for local SDK use.
- The DAG is independently auditable before the ranking is trusted.

### Job Lifecycle

```
submit → queued → causal_phase → pagerank_phase → counterfactual_phase → ready
                                                                        └── error (any phase)
```

State is persisted in PostgreSQL. Clients receive a `job_id` on submission and can poll or register a webhook.

---

## 4. Module Reference

### `CausalEdgeWeighter`

Orchestrates causal discovery. Accepts a raw graph and returns an annotated DAG.

```python
from network_influence.causal import CausalEdgeWeighter

weighter = CausalEdgeWeighter(algorithm="pc")  # or "granger"
dag = weighter.fit(graph)

# dag.edges contains: source, target, weight, p_value, effect_size
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `algorithm` | `str` | `"pc"` | `"pc"` for cross-sectional, `"granger"` for temporal |
| `alpha` | `float` | `0.05` | Significance threshold for edge retention |
| `max_cond_set` | `int` | `4` | PC Algorithm: max conditioning set size |
| `granger_lag` | `int` | `1` | Granger: number of time lags |

---

### `BootstrapCIEngine`

Resamples the graph N times to produce confidence intervals per edge weight and per node score.

```python
from network_influence.bootstrap import BootstrapCIEngine

engine = BootstrapCIEngine(n_iterations=1000, ci_level=0.95)
ci_dag = engine.compute(dag)

# ci_dag.edges[i].ci_lower, ci_dag.edges[i].ci_upper
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_iterations` | `int` | `1000` | Bootstrap resample count |
| `ci_level` | `float` | `0.95` | Confidence interval coverage |
| `seed` | `int` | `42` | Random seed for reproducibility |

---

### `CausalPageRankSolver`

Builds the causal transition matrix and iterates to convergence.

```python
from network_influence.pagerank import CausalPageRankSolver

solver = CausalPageRankSolver(alpha=0.85, epsilon=1e-6)
ranking = solver.solve(ci_dag)

# ranking.nodes → [{"node_id": ..., "score": ..., "ci_lower": ..., "ci_upper": ...}]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.85` | Damping factor |
| `epsilon` | `float` | `1e-6` | Convergence threshold |
| `max_iter` | `int` | `500` | Maximum iterations before forced stop |

---

### `CounterfactualScorer`

Applies do-calculus interventions on the stored DAG.

```python
from network_influence.counterfactual import CounterfactualScorer

scorer = CounterfactualScorer(dag_id="dag_abc123", neo4j_uri=NEO4J_URI)
impact = scorer.removal_impact(node_id="node_42")

# impact.removal_impact_pct → float
# impact.delta_scores       → {node_id: delta, ...}
```

---

### `DAGAuditExporter`

Exports the certified DAG with full provenance metadata.

```python
from network_influence.export import DAGAuditExporter

exporter = DAGAuditExporter(dag)
exporter.to_graphml("output.graphml")
exporter.to_json_ld("output.jsonld")
exporter.to_dot("output.dot")

# Metadata attached to every export:
# algorithm, version, timestamp, sha256_input_hash
```

---

### `SDKCore`

Pure Python module — no Neo4j, no Celery, no PostgreSQL. Intended for local use and WASM compilation.

```python
from network_influence.sdk import SDKCore

sdk = SDKCore()
result = sdk.run(
    edges=[("A", "B", {"time_series": [...]}), ...],
    algorithm="granger",
    top_k=10
)
# result.ranking, result.dag, result.counterfactuals
```

WASM usage (browser/edge via Pyodide):

```js
const { loadPyodide } = require("pyodide");
const pyodide = await loadPyodide();
await pyodide.loadPackage("network-influence-sdk");

const result = await pyodide.runPythonAsync(`
  from network_influence.sdk import SDKCore
  sdk = SDKCore()
  sdk.run(edges=..., algorithm="pc", top_k=5)
`);
```

---

### `JobOrchestrator`

Manages async job lifecycle via Celery + PostgreSQL.

```
POST /v1/jobs          → submit graph, receive job_id
GET  /v1/jobs/{id}     → poll status and result
POST /v1/jobs/{id}/webhook → register callback URL
```

---

## 5. API Reference

Base URL: `https://api.networkinfluence.io/v1`

All requests require `Authorization: Bearer <token>` header.

---

### `POST /v1/analyze`

Submit a graph for causal influence analysis.

**Request body:**

```json
{
  "graph": {
    "format": "edge_list",
    "edges": [
      {"source": "A", "target": "B"},
      {"source": "B", "target": "C"}
    ],
    "node_features": {
      "A": {"time_series": [1.2, 1.5, 1.1]},
      "B": {"time_series": [2.0, 1.8, 2.3]}
    }
  },
  "config": {
    "algorithm": "granger",
    "granger_lag": 2,
    "alpha_significance": 0.05,
    "pagerank_alpha": 0.85,
    "pagerank_epsilon": 1e-6,
    "bootstrap_iterations": 1000,
    "top_k": 10,
    "export_formats": ["graphml", "json_ld"]
  },
  "webhook_url": "https://your-service.com/hooks/influence"
}
```

**Response `202 Accepted`:**

```json
{
  "job_id": "job_7f3a1c",
  "status": "queued",
  "estimated_duration_seconds": 45,
  "polling_url": "/v1/jobs/job_7f3a1c"
}
```

---

### `GET /v1/jobs/{job_id}`

Poll job status and retrieve results when ready.

**Response `200 OK` (status: ready):**

```json
{
  "job_id": "job_7f3a1c",
  "status": "ready",
  "phases": {
    "causal_phase":        "completed",
    "pagerank_phase":      "completed",
    "counterfactual_phase":"completed"
  },
  "result": { ... },
  "dag_id": "dag_abc123",
  "audit": { ... }
}
```

See [Output Schema](#8-output-schema) for the full `result` and `audit` structure.

---

### `GET /v1/dags/{dag_id}`

Retrieve a previously computed causal DAG. Certified DAGs are cached and reusable across jobs.

**Response `200 OK`:**

```json
{
  "dag_id": "dag_abc123",
  "algorithm": "granger",
  "created_at": "2024-11-01T12:00:00Z",
  "sha256_input_hash": "e3b0c44298fc...",
  "edges": [
    {
      "source": "A",
      "target": "B",
      "weight": 0.72,
      "p_value": 0.003,
      "ci_lower": 0.61,
      "ci_upper": 0.84
    }
  ]
}
```

---

### `POST /v1/dags/{dag_id}/counterfactual`

Run a counterfactual intervention on an existing DAG without recomputing causal inference.

**Request body:**

```json
{
  "intervention": "removal",
  "node_id": "node_42"
}
```

**Response `200 OK`:**

```json
{
  "node_id": "node_42",
  "intervention": "do(node_42 = 0)",
  "removal_impact_pct": -34.7,
  "delta_scores": {
    "node_17": -0.082,
    "node_05": -0.041,
    "node_33": -0.019
  }
}
```

---

### `GET /v1/dags/{dag_id}/export/{format}`

Download