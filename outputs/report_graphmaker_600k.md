# Run report

## Transfer (TSTR) — PR-AUC

- **transfer_json**: `outputs/transfer_hi_small_n518573_e600000_m600000_graphmaker_dp.json`
- **device**: `mps`
- **slice**: 600000 tx, 600000 edges, 518573 nodes

| Task | Real → Real | Synthetic → Real |
| ---- | -----------: | ----------------: |
| Node (account fraud) | **0.0022** | **0.0015** |
| Edge (transaction fraud) | **0.3811** | **0.0007** |

## DP accounting (RDP ε)

| Phase | ε (RDPAccountant) |
| ----- | -----------------: |
| node_real | 1.998 |
| edge_real | 1.853 |
| node_syn | 1907.697 |
| edge_syn | 1458.717 |

## PGB-style structural evaluation (GraphMaker)

- **pgb_json**: `outputs/pgb_style_n518573_e600000_m600000.json`

| Metric | Value |
| ------ | ----: |
| Mean scalar RE (capped) | 0.500 |
| Q6 KS statistic | 0.228 |
| Q9 shortest-path L1 | 1.000 |
| Q12 community NMI | — |
| Directed edge Jaccard | 0.000 |
| Aligned node-feature mean L1 | — |
