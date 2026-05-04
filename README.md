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

### Reference numbers (900-node GraphMaker demo, 20k balanced edges)

From `outputs/run/transfer_hi_small_n518573_e20000_m20000_graphmaker.json` (with `amlworld_graph.bin`, `amlworld_cpts/`, and `synthetic_from_graphmaker.pt` in the same folder):

**Sample sizes (real)**  
- Nodes (full graph): 518,573; **node test:** 103,715 (**6** pos).  
- **Edges:** 20,000 loaded; **edge test:** 4,000 (**5** pos).  
- Stratified **60/20/20** edges when counts allow (`split_edge_policy: stratified_60_20_20_sklearn`).

**PR-AUC (illustrative)**

| Task | Real → real (test) | Synthetic → real (test) |
|------|-------------------:|--------------------------:|
| Node | 0.00156 | 0.000097 |
| Edge | 1.000 | 0.00077 |

Edge real test PR-AUC = 1 with **5** positives is still noisy; the contrast with transfer is the main story for this demo.

## 4. Other entry points (short)

| Script | Role |
|--------|------|
| `scripts/run_experiment.py` | Transfer only: point `--synthetic_pt` at an existing `.pt`, or `--generator degree_preserving` for a non-GraphMaker baseline. |
| `scripts/graphmaker_train_amlworld.py` | Train only (writes `amlworld_graph.bin` + copies checkpoints). Respects same fast env defaults unless `GRAPHMAKER_USE_YAML_DEFAULTS=1`. |
| `scripts/graphmaker_sample_to_pyg.py` | Sample only (needs `AMLWORLD_DGL_GRAPH` + checkpoint). |

**Optional:** differential privacy and edges-only training paths still exist in `models/dp_train.py` and `scripts/run_experiment.py` (`--help`); they are not required for the GraphMaker pipeline above.

## 5. Code map

| Path | Role |
|------|------|
| `data/amlworld.py` | CSV → PyG; **`slice_mode`**, **`balanced_edges`**, stratified splits when viable. |
| `generators/graphmaker_aml/build.py` | PyG → DGL for GraphMaker; induced subgraph. |
| `third_party/GraphMaker/train_amlworld_async.py` | Training loop; env overrides for fast runs. |
| `pipeline/transfer_experiment.py` | Real / synthetic training and **transfer** evaluation. |
