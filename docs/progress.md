# SmartGC Progress & Milestones

## Phase Status Summary

| Phase | Description | Status | Commit / Milestone Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton, Git init | **In Progress** | Repository initialized, contracts documented, CMake build configured |
| **Phase 1** | Baseline LFS Simulator + Greedy GC (C++17), unit tests, synthetic workload, WAF verification | **In Progress** | Core simulator, block/segment model, L2P table, Greedy GC, test harness |
| **Phase 2** | Dataset inspection & raw trace normalization (Python) | **Pending** | Awaiting Phase 1 completion and user go-ahead |
| **Phase 3** | Historical rewrite-interval sequence construction & split (Python) | **Pending** | Awaiting Phase 2 |
| **Phase 4** | PyTorch lightweight LSTM training, evaluation & thresholding | **Pending** | Awaiting Phase 3 |
| **Phase 5** | Rule-based heuristic & LSTM prediction integration into simulator | **Pending** | Awaiting Phase 4 |
| **Phase 6** | Controlled multi-policy benchmark execution & raw metrics collection | **Pending** | Awaiting Phase 5 |
| **Phase 7** | Visualization plots & comprehensive comparative analysis writeup | **Pending** | Awaiting Phase 6 |

---

## Decisions Log

1. **Decoupled Architecture**: C++ and Python subsystems interact exclusively through CSV contract files to eliminate binding dependencies and runtime complexity.
2. **Fixed Random Seeds**: All random generators (workloads, ML data split, model init) enforce explicit, logged seeds for bitwise reproducibility.
3. **Strict Novelty Boundary**: Hot/cold separation is documented as an established file system concept; SmartGC's scope is strictly evaluating sequence-based deep learning interval prediction vs. baseline heuristics in a controlled simulator.
4. **WAF Definition**: $\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$. GC migration writes contribute solely to physical writes.
