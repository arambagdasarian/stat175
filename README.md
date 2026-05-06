# AMLWorld → GraphMaker → Transfer (quick pipeline)

This repo builds a **PyTorch Geometric** transaction graph from **AMLWorld HI-Small**, optionally **rebalances** rare fraud labels, trains **GraphMaker-Async** on a **small induced subgraph** (fast), exports a synthetic **`.pt`**, and runs a **transfer experiment**: train GraphSAGE + heads on real data, then on synthetic data, then **evaluate on the real test split**.

## 1. Data and install

1. Put **`HI-Small_Trans.csv`** and **`HI-Small_accounts.csv`** in `data/raw/` ([Kaggle AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)). If missing, `scripts/run_experiment.py` can try a one-time Kaggle CLI download when `kaggle` is configured.
2. **Base install:** `pip install -e .`
3. **GraphMaker (DGL):** `pip install -e ".[graphmaker]"` — DGL must match your PyTorch build ([DGL start](https://www.dgl.ai/pages/start.html)).

## 2. Main command: end-to-end GraphMaker pipeline

### 2a. Default smoke run (1000-node GraphMaker slice)

From the **repository root** (directory that contains `pyproject.toml`):

```bash
WANDB_MODE=disabled python3 -m scripts.run_graphmaker_pipeline --data_dir data/raw
```

This writes under **`outputs/graphmaker/`** by default (GraphMaker graph, checkpoint copy, synthetic `.pt`, and `transfer_*.json` next to them).

| Stage | What happens |
|-------|----------------|
| **Build DGL graph** | Load CSV → PyG → **random `max_nodes` subset** (default 1000) for GraphMaker only (`generators/graphmaker_aml/build.py`). |
| **Train GraphMaker** | Short epochs + large minibatches via env (see `scripts/run_graphmaker_pipeline.py`); upstream trains over **all upper-triangular node pairs** each epoch → cost is **Θ(N²)** in `max_nodes`. |
| **Sample → PyG** | `scripts/graphmaker_sample_to_pyg.py` writes `synthetic_from_graphmaker.pt`. |
| **Transfer** | `scripts/run_experiment.py` trains/evaluates on the **full** real PyG slice (same `max_transactions` / slice flags), not the tiny DGL node set. |

### 2b. 900-node run with balanced edges (recommended preset)

Use this when you want **more laundering edges in the real slice** and a **900-node** GraphMaker training graph (same setup as the reference numbers in §3).

1. **Go to repo root** (example on macOS if the project lives in iCloud; quote paths with spaces):

   ```bash
   cd "/path/to/Final Project"
   ```

2. **(Optional) Virtualenv:** `python3 -m venv .venv && source .venv/bin/activate`

3. **Install** (once per environment): `pip install -e .` then `pip install -e ".[graphmaker]"` as in §1.

4. **Data files:** ensure these exist relative to the repo root:

   - `data/raw/HI-Small_Trans.csv`
   - `data/raw/HI-Small_accounts.csv`

5. **Sanity-check DGL / GraphMaker** (should print `torch …`, `dgl …`, and `GraphBolt path OK`):

   ```bash
   WANDB_MODE=disabled python3 -m scripts.verify_graphmaker_env
   ```

6. **Run the full pipeline** (trains GraphMaker on **900** nodes, builds a **20k-edge** balanced real slice, writes everything under **`outputs/run/`** — that folder is overwritten each time):

   ```bash
   WANDB_MODE=disabled python3 -m scripts.run_graphmaker_pipeline \
     --data_dir data/raw \
     --out_dir outputs/run \
     --max_transactions 20000 \
     --max_nodes 900 \
     --slice_mode balanced_edges \
     --balance_scan_rows 250000 \
     --target_edge_pos_fraction 0.18 \
     --syn_edge_fraud_rate 0.22
   ```

7. **Check it worked:** the process exits with code **0**, and you should see a final line like  
   `Wrote results to outputs/run/transfer_hi_small_n518573_e20000_m20000_graphmaker.json`.

8. **Artifacts in `outputs/run/`** after a successful run:

   | File / directory | Role |
   |------------------|------|
   | `amlworld_graph.bin` | DGL graph used for GraphMaker (900-node induced subgraph). |
   | `amlworld_cpts/` | Copied checkpoint (e.g. `Async_TX6_TE9.pth`) used for sampling. |
   | `synthetic_from_graphmaker.pt` | PyG synthetic graph for transfer. |
   | `transfer_hi_small_n518573_e20000_m20000_graphmaker.json` | Metrics, splits, similarity diagnostics. |

9. **Rough runtime:** often **about one to a few minutes** on a recent laptop (CSV scan + short GraphMaker epochs + short GNN transfer); CPU vs MPS/CUDA and disk speed dominate.

**What the flags do**

- **`--slice_mode balanced_edges`**: scan up to `balance_scan_rows`, then subsample to `max_transactions` edges with about `target_edge_pos_fraction` laundering edges (capped by positives in the window). Improves **train/val/test positive counts** on real edges so PR-AUC is interpretable.
- **`--syn_edge_fraud_rate`**: Bernoulli rate for **synthetic** edge labels (GraphMaker does not recover real fraud labels); set high enough that synthetic **edge** splits contain positives.
- **`--max_nodes`**: smaller ⇒ faster GraphMaker; **`min_synthetic_nodes`** defaults to `min(8192, max_nodes)` so transfer does not require 8k nodes when the training graph is tiny.

**Slower / YAML-faithful GraphMaker:** `--full_graphmaker_train` (200 epochs, original batch sizes — not for CPU). **Longer GNN transfer:** `--full_transfer` (drops the short demo `--node_epochs` / `--edge_epochs` overrides).

## 3. How to read the result JSON

By default, `run_graphmaker_pipeline` writes **`transfer_hi_small_n<N>_e<E>_m<M>_graphmaker.json`** in the same directory as `--out_dir` (override with `--experiment_out_dir` if you want JSON elsewhere). The checked-in reference run is **`outputs/run/`** (GraphMaker files + that JSON).

| Block | Meaning |
|-------|---------|
| `dataset_meta` | Slice (`slice_mode`, `balance_scan_rows`, …), **`split_*_label_counts`** (positives per split — check these before trusting AUC). |
| `result.node` | Real train/val/test vs **transfer** (encoder trained on synthetic, evaluated on **real** node test mask). |
| `result.edge` | Same for edges; **`synthetic_train_eval`** shows whether the synthetic graph had positives in each split. |
| `result.similarity` | Structural / feature alignment diagnostics (degree, path length, KS on features). |

**Interpretation:** if real **test** has only a handful of positives, ROC can swing; **PR-AUC** plus **`n_pos` / `n_neg`** in each row is the honest read. Large **drop_test** between real-trained and transfer on the same real test fold means synthetic pre-training did not substitute for real labels under that synthetic process.

### Reference numbers

All experiments use the same real slice: **200k balanced edges** (1M CSV row scan, 6% target fraud fraction), 518,573 nodes, stratified 60/20/20 split → **115 positive edges** and **151 positive nodes** in the real test set.

#### Summary comparison

| Experiment | Synthetic type | Edge transfer PR-AUC | Edge PR-lift | Node transfer PR-AUC |
|---|---|---:|---:|---:|
| **Real trained (ceiling)** | — | **0.449** | 156× | 0.00485 |
| **C — Degree-preserving** | Full-scale synthetic | **0.458** | **159×** | 0.00230 |
| **B — Snowball subgraph** | Real 1,200-node slice | 0.0665 | 23× | 0.00162 |
| **A — GraphMaker (random)** | Generative model | 0.00077 | 0.6× | 0.000097 |

PR-lift = PR-AUC / prevalence (0.29%). Edge PR-AUC is the primary metric; node results are noisier due to class imbalance (151 positives / 103k test nodes).

---

#### Experiment A — GraphMaker on random-induced subgraph (20k edges, 900 nodes)

From `outputs/run/transfer_hi_small_n518573_e20000_m20000_graphmaker.json`.

**Edge test: 4,000 edges, 5 positives.** The random-induced 900-node subgraph contained 0 SAR nodes and 0 fraud edges — GraphMaker had no laundering signal to learn from, and the edge transfer PR-AUC of 0.00077 is below random (PR-lift 0.6×).

---

#### Experiment B — Snowball subgraph (real data, 200k edges)

From `outputs/snowball_run/transfer_hi_small_n518573_e200000_m200000_snowball.json`.

The GNN is trained directly on a **snowball-sampled subgraph** seeded from the 20 highest-out-degree SAR accounts and expanded one wave until the 1,200-node budget is filled. This is **real data**, not a generative model.

**Subgraph quality vs random induced:**

| | Random induced (Exp A) | Snowball (Exp B) |
|---|---:|---:|
| Nodes | 900 | 1,200 |
| Edges | ~0 fraud | 1,637 (5.1% fraud) |
| SAR nodes | 0 (0%) | 106 (8.8%) |
| Isolated nodes | ~80% | 0.8% |

**Limitations:** Snowball is real data, so 0.0665 is a ceiling for what a generative model on this subgraph could achieve. The 7× gap vs the real-trained ceiling (0.449) shows that 1,200 nodes are too few to cover the full graph's structural diversity. Node-level transfer is weak because the subgraph's local hub structure doesn't generalise to the full 518k-node graph.

**To reproduce:**

```bash
python3 -m scripts.run_experiment \
  --data_dir data/raw \
  --slice_mode balanced_edges \
  --max_transactions 200000 \
  --balance_scan_rows 1000000 \
  --target_edge_pos_fraction 0.06 \
  --generator snowball \
  --snowball_top_k 20 \
  --snowball_max_nodes 1200 \
  --out_dir outputs/snowball_run \
  --seed 7
```

---

#### Experiment C — Degree-preserving baseline (200k edges)

From `outputs/degree_preserving_run/transfer_hi_small_n518573_e200000_m200000_degree_preserving.json`.

Generates a fully synthetic graph by **preserving the exact in- and out-degree sequence** of the full real graph (stub matching) and independently bootstrapping node features, edge features, and labels from their empirical marginal distributions (`generators/degree_preserving.py`).

**Key result:** transfer edge PR-AUC of **0.458** — essentially matching the real-trained ceiling of 0.449 (drop of −0.009, i.e. slightly *better*, within noise). PR-lift is **159×** vs 156× for real-trained.

**Structural fidelity:** degree sequence L1 error = 0 (exact match by construction); node-feature KS mean = 0.00052; edge-feature KS mean = 0.0011; fraud label rates match to within 0.01%.

**Limitation:** degree-preserving does **not** model the joint distribution of degree and fraud label. In the real graph, SAR nodes have mean out-degree 14.7; in the synthetic graph, SAR nodes have mean out-degree 0.36 (same as clean nodes, because labels are assigned from the marginal, ignoring topology). The model still transfers well for edge classification because preserving the degree sequence is sufficient to reproduce the graph topology that the GNN uses — but the synthetic graph cannot faithfully reproduce the hub-and-spoke laundering motifs that distinguish fraud accounts.

**To reproduce:**

```bash
python3 -m scripts.run_experiment \
  --data_dir data/raw \
  --slice_mode balanced_edges \
  --max_transactions 200000 \
  --balance_scan_rows 1000000 \
  --target_edge_pos_fraction 0.06 \
  --generator degree_preserving \
  --out_dir outputs/degree_preserving_run \
  --seed 7
```

## 4. Other entry points (short)

| Script | Role |
|--------|------|
| `scripts/run_experiment.py` | Transfer only: `--generator from_pt` (GraphMaker `.pt`), `--generator degree_preserving` (degree-sequence baseline), or `--generator snowball` (fraud-hub subgraph — Exp B above). |
| `scripts/graphmaker_train_amlworld.py` | Train only (writes `amlworld_graph.bin` + copies checkpoints). Respects same fast env defaults unless `GRAPHMAKER_USE_YAML_DEFAULTS=1`. |
| `scripts/graphmaker_sample_to_pyg.py` | Sample only (needs `AMLWORLD_DGL_GRAPH` + checkpoint). |

**Optional:** differential privacy and edges-only training paths still exist in `models/dp_train.py` and `scripts/run_experiment.py` (`--help`); they are not required for the GraphMaker pipeline above.

## 5. Code map

| Path | Role |
|------|------|
| `data/amlworld.py` | CSV → PyG; **`slice_mode`**, **`balanced_edges`**, stratified splits when viable. |
| `generators/graphmaker_aml/build.py` | PyG → DGL for GraphMaker; supports `--sampling snowball/fraud_enriched/random_induced`. |
| `generators/snowball_sampling.py` | Wave-by-wave snowball expansion from top-k SAR hubs; used by `--generator snowball`. |
| `scripts/snowball_stats.py` | Numpy-only diagnostic: compares random-induced vs snowball subgraph stats (no torch needed). |
| `third_party/GraphMaker/train_amlworld_async.py` | Training loop; env overrides for fast runs. |
| `pipeline/transfer_experiment.py` | Real / synthetic training and **transfer** evaluation. |
