# Run report

## Transfer (TSTR) — PR-AUC

- **transfer_json**: `outputs/transfer_hi_small_n518573_e300000_m300000_graphmaker_dp.json`
- **device**: `cpu`
- **slice**: 300000 tx, 300000 edges, 518573 nodes

| Task | Real → Real | Synthetic → Real |
| ---- | -----------: | ----------------: |
| Node (account fraud) | **0.0004** | **0.0001** |
| Edge (transaction fraud) | **0.4636** | **0.0003** |

## DP accounting (RDP ε)

| Phase | ε (RDPAccountant) |
| ----- | -----------------: |
| node_real | 0.749 |
| edge_real | 0.854 |
| node_syn | 342.861 |
| edge_syn | 192.036 |

## PGB-style structural evaluation (GraphMaker)

- **pgb_json**: `outputs/pgb_style_n518573_e300000_m300000.json`

| Metric | Value |
| ------ | ----: |
| Mean scalar RE (capped) | 0.500 |
| Q6 KS statistic | 0.139 |
| Q9 shortest-path L1 | 1.000 |
| Q12 community NMI | — |
| Directed edge Jaccard | 0.000 |
| Aligned node-feature mean L1 | — |
