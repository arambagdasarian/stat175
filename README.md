# AMLWorld Synthetic Graphs — Evaluating Performance of Anonymized Data

## Quick start

1. **Data:** put `HI-Small_Trans.csv` and `HI-Small_accounts.csv` in `data/raw/` (from [Kaggle AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)). If they are missing, `scripts/run_experiment.py` tries a one-time Kaggle CLI download when `kaggle` is configured.
2. **Install:** `pip install -e .` — for DP runs also: `pip install -e ".[dp]"`.
3. **Transfer experiment (default: first 600k tx rows, GraphMaker `.pt`, DP):**  
   `python3 -m scripts.run_experiment --use_dp --out_dir outputs`  
   Faster smoke (no GraphMaker file):  
   `python3 -m scripts.run_experiment --generator degree_preserving --use_dp --max_transactions 50000 --out_dir outputs`
4. **Markdown report from a result JSON:**  
   `python3 -m scripts.write_md_report --transfer_json outputs/<run>.json --out_md outputs/report.md`  
   Optional PGB block: add `--pgb_json outputs/pgb_style_....json`.

## 1. Introduction and Motivation

The application of machine learning to anti–money laundering (AML) has introduced a fundamental challenge: the most valuable datasets for model development are also the most restricted. Financial transaction data is highly sensitive, and regulatory constraints limit its availability for external analysis, benchmarking, and model development.

Synthetic data generation offers a principled approach to addressing this constraint. Instead of sharing real transaction data, a financial institution may release a synthetic graph that approximates the statistical and structural properties of the original dataset. The central question, however, is not whether the synthetic data “looks similar,” but whether it preserves the information necessary for downstream tasks.

This project investigates the following problem:

Can a model trained on a synthetic transaction graph achieve comparable performance when evaluated on the original real transaction graph?

To answer this, we construct an experimental framework centered on anonymized graphs. Specifically, we compare model performance when trained on real data versus synthetic data, holding the evaluation dataset fixed. This provides a task-driven measure of synthetic data quality.

## 2. Data and Graph Representation

The experiments are conducted using the AMLWorld HI-Small dataset, which models financial activity as a directed transaction graph.

- Nodes represent accounts  
- Edges represent transactions between accounts

Each transaction includes attributes such as amount, timestamp, and payment format, along with labels indicating whether it is associated with laundering activity.

The dataset is converted into a graph representation using PyTorch Geometric. This involves:

- Constructing a directed edge index from sender to receiver  
- Defining node features (e.g., bank identity, entity type)  
- Defining edge features (e.g., log-transformed amount, relative time, payment type)

The resulting graph captures both relational structure and transactional attributes, enabling the application of graph-based learning methods.

## 3. Fraud Detection Model

### 3.1 GraphSAGE for Node Representation

Fraud detection in transaction networks is inherently relational, as suspicious behavior often arises from patterns of interaction rather than isolated events. To capture these dependencies, we employ GraphSAGE, a graph neural network that learns node representations through neighborhood aggregation.

At each layer, a node updates its representation by combining its own features with an aggregate of its neighbors’ features. Stacking multiple layers allows the model to incorporate information from progressively larger neighborhoods. A linear classifier is then applied to these embeddings to predict whether an account is associated with fraudulent activity.

### 3.2 Edge Classification

In addition to node-level inference, the model performs edge classification. For each transaction, a feature vector is constructed by concatenating:

- The embedding of the source node  
- The embedding of the destination node  
- The edge feature vector

A multilayer perceptron is then used to classify whether the transaction is fraudulent. This setup allows the model to use both the surrounding graph structure and the transaction’s own attributes.

### 3.3 Edges-only (transaction-focused) training

By default, the transfer pipeline first trains a **node classifier** (account labels), then an **edge classifier** on top of the same GraphSAGE encoder. For settings where the node task is uninformative or too costly, you can run `**--edges_only`** (`scripts/run_experiment.py` → `ModelConfig.edges_only` in `pipeline/transfer_experiment.py`):

- **Skipped:** node training on real and synthetic, and node transfer evaluation on the real test split. The JSON `result.node` records `skipped: true` and a short reason instead of metrics.
- **Still run:** joint training of the **GraphSAGE encoder and edge MLP** on the edge objective (real graph, then synthetic graph), edge transfer evaluation on the real test edges, and the usual structural / fraud-pattern reports.
- **DP accounting:** with `--use_dp` and `--edges_only`, RDP metadata appears only for `**edge_real`** and `**edge_syn**` (no separate node DP phases). The reported `dp_composition_note` reflects that.

**Checkpoints (DP training):** pass `**--checkpoint_dir PATH`** together with `**--use_dp**`. Each DP phase writes per-epoch weights and a `*_best.pt` when validation PR-AUC improves: `**node_real_***`, `**edge_real_***`, `**node_syn_***`, `**edge_syn_***` (edge checkpoints bundle encoder + edge head; see `models/dp_train.py`). With `**--edges_only**`, only the `edge_*` files are written. Result JSON includes `**checkpoint_dir**` (and legacy `**edge_checkpoint_dir**`, same path) plus per-phase `checkpoints_written` in `dp_accounting`.

Output filenames append `**_edgesonly**` and `**_dp**` when those flags are used (e.g. `…_degree_preserving_edgesonly_dp.json`).

## 4. Synthetic Graph Generation

The project evaluates two approaches to synthetic graph generation.

### 4.1 Degree-Preserving Baseline

The baseline generator preserves the in-degree and out-degree sequences of the original graph. Edges are reassigned randomly while maintaining these degree constraints. Node and edge features are sampled independently from empirical distributions.

This method keeps basic graph statistics intact but does not attempt to reproduce more complex patterns like clusters or repeated transaction motifs.

### 4.2 GraphMaker

GraphMaker is a diffusion-based model for generating graphs with attributes. The idea is to learn how to reconstruct a graph after it has been gradually corrupted with noise.

During training, the model observes increasingly noisy versions of the original graph and learns how to reverse that process. At generation time, it starts from noise and iteratively builds a graph.

In practice, GraphMaker can first generate node attributes and then generate edges based on those attributes. This is useful for financial data, where the type of account often influences how it interacts with others.

Compared to the baseline, GraphMaker is designed to capture more complex structure in the data rather than just matching simple statistics.

## 5. Experimental Framework

The pipeline follows a consistent structure:

1. Build a graph from AMLWorld data
2. Train a GraphSAGE-based model on the real graph (**DP-SGD**: random *k*-hop subgraph minibatches, gradient clipping, Gaussian noise; see `models/dp_train.py`). With `**--edges_only`**, this step omits the node classifier and trains the encoder **only** through the edge objective.
3. Generate a synthetic graph
4. Train the same model on the synthetic graph (same DP-SGD recipe; again node training is omitted when `**--edges_only`**)
5. Evaluate both models on the real test set (node metrics omitted in JSON when edges-only)

The main comparison is between:

- Training on real data and testing on real data  
- Training on synthetic data and testing on real data

This gives a direct sense of how useful the synthetic data is for the actual task.

Because fraud is extremely rare, a few practical adjustments are needed:

- The loss function is weighted to account for imbalance  
- Extremely large class weights are capped to avoid unstable training  
- Training schedules are slightly adjusted to ensure convergence  
- Dropout is used for regularization

## 6. Results

This section has two parts: **(1)** GraphMaker transfer metrics on the held-out **real** test split—**ROC-AUC** and **PR-AUC** (PR-AUC tracks rare positives more directly; node fraud is so sparse that node PR-AUC stays small even when ROC-AUC is moderate) and **(2)** **PGB-style structural fidelity** (GraphMaker column only—same 300k slice, `outputs/pgb_style_n518573_e300000_m300000.json`).

Experiments use **differentially private training** (`--use_dp`), full pipeline (node then edge on real, then synthetic), **`--generator from_pt`**, and the **first 300,000** HI-Small_Trans rows (`max_transactions_loaded: 300000`). GraphMaker weights are **lazy-loaded after real training**; DP training uses **balanced pos/neg minibatches** (`models/dp_train.py`, `pipeline/transfer_experiment.py`). Source JSON: `outputs/transfer_hi_small_n518573_e300000_m300000_graphmaker_dp.json`.

### GraphMaker transfer (test ROC-AUC)

Held-out **real** test nodes (account task) and **60,000** real test edges (transaction task).

| Task | Real → Real | Synthetic → Real |
| ---- | -----------: | ----------------: |
| Node (account fraud) | **0.821** | **0.464** |
| Edge (transaction fraud) | **0.765** | **0.406** |

### GraphMaker transfer (test PR-AUC)

Same folds as above. **Four decimals** on PR-AUC match `scripts/write_md_report.py`.

| Task | Real → Real | Synthetic → Real |
| ---- | -----------: | ----------------: |
| Node (account fraud) | **0.0004** | **0.0001** |
| Edge (transaction fraud) | **0.4636** | **0.0003** |

Synthetic→Real uses the encoder trained on the **GraphMaker** `.pt` graph and evaluated on the **same** real test masks / edge indices; the checked-in synthetic sample is small/degenerate, so transfer metrics drop versus real training.

**Scale (not ~5M):** **300,000** transactions → **300,000** edges, **518,573** nodes; **60% / 20% / 20%** splits on nodes and edges.

**We did not run this table on the full ~5M-row CSV.** Use `--all_transactions --use_dp --generator from_pt` (plus paths) and refresh from the new JSON.

### Edges-only DP (optional)

Small **illustrative** run for **`--edges_only`** + checkpoints: see `outputs/transfer_hi_small_n518573_e5000_m5000_degree_preserving_edgesonly_dp.json` and `dp_accounting` therein (not the main results above).

### PGB-style structural evaluation (Liu et al., arXiv:2408.02928)

We report **structural fidelity** diagnostics in `evaluation/pgb_style.py` (undirected simple projections; **not** an ε guarantee from GraphMaker). Same **300,000**-transaction real slice as above. Full JSON: `outputs/pgb_style_n518573_e300000_m300000.json` (regenerate with `python3 -m scripts.run_pgb_style --max_transactions 300000 --out_dir outputs`).


| Metric (lower is often better for distances / RE) | GraphMaker |
| ------------------------------------------------- | ---------: |
| Mean scalar relative error (capped)¹              | 0.500      |
| Q6 — degree distribution (KS statistic)           | 0.139      |
| Q9 — shortest-path histogram L1                   | 1.000      |
| Q12 — community agreement (NMI)                   | —²         |
| Q11 — attribute correlation / MRE (Table IV)    | 0.139      |
| Q15 — eigenvector centrality MAE                  | —⁴         |
| Directed edge-set Jaccard (real vs syn)           | 0.000      |
| Aligned node-feature mean L1                      | —³         |


¹ Mean of finite relative errors for scalar graph queries (|V|, |E|, triangles, degrees, diameter, etc.); see JSON `summary_mean_RE_scalar_queries`.  
² Skipped when the largest connected components are too large for greedy modularity or the GraphMaker graph is degenerate.  
³ GraphMaker export has **800** nodes vs **518,573** on the real graph, so feature vectors are not row-aligned.  
⁴ GraphMaker row: eigenvector centrality query **skipped** in JSON (`GCC intersection empty or too large for EVC`).

**Note:** The checked-in `outputs/graphmaker/synthetic_from_graphmaker.pt` currently contains **only a self-loop** on one node (no usable undirected edges), so the GraphMaker PGB row mainly reflects that degenerate sample. Re-sample from GraphMaker to obtain a dense synthetic graph before reading much into those numbers.

## 7. Discussion

On the 300k slice with **balanced DP minibatches**, **GraphMaker Real → Real** is strong on **ROC-AUC** for node and edge (**~0.82** / **~0.77**), while **Synthetic → Real** falls on **ROC-AUC** and especially on **PR-AUC** (edge PR-AUC near baseline; node PR-AUC tiny under label sparsity) because the checked-in GraphMaker export is tiny/degenerate, not because the real phase is mis-specified.

**Edges-only** runs skip the node classifier and halve DP bookkeeping to edge phases only; see §6 optional pointer.

Treat the GraphMaker **PGB** row as structural diagnostics on that same export, not a claim about a full-scale synthetic AML graph until you re-sample a denser graph.

## 8. Code layout and longer runs

| Path | Role |
|------|------|
| `data/amlworld.py` | CSV → PyG `Data` |
| `generators/` | Degree-preserving baseline; GraphMaker bridge |
| `models/` | GraphSAGE, `train_utils.py`, **DP-SGD** in `dp_train.py` |
| `pipeline/transfer_experiment.py` | End-to-end transfer; optional **`--edges_only`** |
| `evaluation/` | Similarity, **PGB-style** (`pgb_style.py`), link leakage |

**Regenerate §6 tables:** match `--max_transactions` / `--generator` to the JSON you care about, then run `scripts/run_pgb_style.py` for PGB. **`opacus`** supplies RDP (ε, δ) in logs (`pip install -e ".[dp]"`).

**Edges-only DP + checkpoints** (encoder + edge head only):

```bash
python3 -m scripts.run_experiment --use_dp --edges_only --generator degree_preserving \
  --checkpoint_dir outputs/my_edge_checkpoints --device mps --out_dir outputs
```

**Full ~5M-row CSV:** `--all_transactions` or `--max_transactions 0`. Expect large RAM and long runs on CPU; use **GPU/MPS** and lower `--dp_steps_per_epoch`, `--node_epochs`, `--edge_epochs` while iterating. PGB caps edges per projection (`--pgb_max_edge_rows`, default 600k).

**Device:** auto-picks CUDA, else MPS, else CPU (`models/torch_device.py`); override with `--device cpu` if you hit backend bugs.

Other utilities: `scripts/visualize_outputs.py`, `scripts/run_link_leakage_audit.py`, GraphMaker train/sample scripts under `scripts/` (see `--help` on each).