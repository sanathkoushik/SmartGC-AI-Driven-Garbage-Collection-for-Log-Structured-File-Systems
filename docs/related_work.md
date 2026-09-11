# Related Work — verified citations

Metadata below was re-checked against the ACM Digital Library / IEEE Xplore /
ScienceDirect in **September 2026**. Where an earlier internal draft of the
scope-restructuring analysis was approximate or wrong, the correction is noted.
These are the nine works the SmartGC scope restructuring is anchored to; each
row also names which SmartGC component it motivates.

| # | Citation (verified) | Venue / date | Motivates |
|---|---|---|---|
| 1 | P. Sun, S. Zheng, L. You, W. Zhang, R. Ma, J. Yang, F. Zhu, S. Li, L. Huang. **"Shiro: Efficient and Accurate In-Storage Data Lifetime Separation for NAND Flash SSDs."** | *IEEE TCAD*, **Feb 2026** (IEEE Xplore doc. 11073159) | Placement-time **and** GC-time learned separation; a *sequence model* over long history predicts continuous lifetime; multi-destination GC migration. → SmartGC's 3-level `StreamClass` migration + `decide_migration_stream()`. |
| 2 | **"iGC: Reinforcement learning-guided intelligent garbage collection strategy for multi-tenant SSDs."** | *Knowledge-Based Systems*, **Dec 2025** (ScienceDirect PII S0950705125020519) | Q-learning decides GC **timing and aggressiveness** from I/O pressure + free-block availability, vs a fixed watermark. → SmartGC's `GcTriggerController` + `ml/training/gc_controller.py`. |
| 3 | H. Sun, H. Ding, H. Tong, X. Cheng, H. Chai, X. Qin. **"AGC: An Adaptive Workload Burst-Aware Garbage Collection Mechanism for High-Performance SSDs."** | *IEEE Transactions on Computers*, **Vol. 75, No. 3 (Mar 2026), pp. 845–859** (IEEE Xplore doc. 11298441) | Detects workload bursts (5 ms windows) and adapts GC scheduling. → SmartGC's `recent_write_rate` trigger-state term + the drift scenario / detector. |
| 4 | **"DumpKV: Learning based lifetime aware garbage collection for key value separation in LSM-tree."** | ***PVLDB* Vol. 18 (2025)**, DOI 10.14778/3717755.3717778 (arXiv:2406.01250, Jun 2024) — *spec said "arXiv 2024" only; it was published at VLDB 2025* | A lightweight per-key model with **dynamically adjusted** lifetime thresholds and a lifetime-bucketed value-file layout; reports 38–73% WA reduction. → SmartGC's rolling-percentile cutoff (`ml/inference/rolling_cutoff.py`) replacing the fixed `hot_percentile_cutoff`. |
| 5 | **"Simplicity as the Ultimate Principle: The Art of Garbage Collection Management in SSDs Inspired by Natural Data Behavior"** (SUP-GC). | *ACM Transactions on Storage*, **19 Mar 2025**, DOI 10.1145/3725219 | Near-zero-computation heuristic: all fresh writes → hot blocks; valid data retained through GC → cold blocks. A deliberately strong "why do you need ML" baseline. → SmartGC's `PlacementPolicy::SUP_LIKE` (implemented exactly: default-hot-on-write, cold-on-GC-copy). |
| 6 | **"Lightweight data lifetime classification using migration counts to improve performance and lifetime of flash-based SSDs"** (MiDA — *Migration count based Data Age classification*). | *APSys* — **12th ACM SIGOPS APSys, 2021**, DOI 10.1145/3476886.3477520 — *spec said "APSys 2025, MDA-style"; the paper is MiDA and from APSys 2021* | Classifies temperature from cheap runtime signals (migration count, recency) at negligible overhead, no trained model. → motivates SmartGC's `RULE_BASED` rung and the migration-count feature intuition. |
| 7 | F. Serajeh Hassani, A. Gheibi-Fetrat, S. Hosseini, S. Lee, H. Sarbazi-Azad. **"Garbage Collection Techniques in Solid-State Drives (SSDs)."** | *ACM Computing Surveys*, **Vol. 58, No. 13 (24 Jun 2026), pp. 1–36**, DOI 10.1145/3816041 | Taxonomy of data-separation and victim-selection strategies; situates SmartGC's greedy victim selection + multi-stream separation in the design space. |
| 8 | J. Wang, Z. Tan, Y. Xiao, W. Qi, S. Tan, Y. Yuan. **"NatSep: Little-to-No Overhead Data Separation for Log-Structured Storage Using Native Information."** | *Proc. 43rd IEEE Int'l Conf. on Computer Design (ICCD)*, **2025**, IEEE Xplore doc. 11310889 | A 2025 near-zero-overhead heuristic for log-structured storage specifically (temperature regions + unidirectional rollback-style writing from *native* signals, no model). A closer, more current point of comparison than MiDA (2021, generic flash) for the ladder's own finding that cheap heuristics beat learned rungs on real OLTP data — motivates treating `RULE_BASED`/`SUP_LIKE` as a serious baseline family, not a strawman, and is a natural future rung to add. |
| 9 | T. Lange, J. (Seffi) Naor, G. Yadgar. **"Optimal SSD Management with Predictions."** | *Proc. ACM Meas. Anal. Comput. Syst. (SIGMETRICS)*, **9(2), Article 30, June 2025**, DOI 10.1145/3727122 | A learning-augmented-algorithms treatment of SSD rewrite minimization: each page write carries a predicted death-time, and the paper's framing centres on a *consistency-robustness* tradeoff -- performance when predictions are accurate vs. a worst-case guarantee when they are wrong. Directly motivates `PlacementPolicy::HYBRID_ROBUST_SMARTGC` (Phase 8, `ml/inference/robust_blend.py`): a continuous, confidence-weighted, robustness-capped blend of the LSTM+attention prediction with `RULE_BASED`, replacing the binary `ConfidenceGate` cutoff with a tunable `robustness_lambda` that generalises it. This is a heuristic instantiation of that framing for an empirical ladder comparison, not a reproduction of the paper's formal competitive-ratio algorithm. |

## Corrections vs. an earlier internal draft of this analysis

- **DumpKV** is not "arXiv, 2024" only — it was published in **PVLDB Vol. 18 (2025)**; cite the VLDB version.
- The migration-count paper is **MiDA** (not "MDA") and is **APSys 2021**, not 2025.
- **Shiro** is now firmly published: *IEEE TCAD*, **Feb 2026** (was "online first").
- **AGC** now has full coordinates: *IEEE TC* **75(3):845–859, Mar 2026**.
- The **ACM CSUR** GC survey is **58(13), Jun 2026**.
- **iGC** (*KBS*, Dec 2025) and **SUP-GC** (*ACM ToS*, Mar 2025) confirmed as originally stated.
- **NatSep** and **"Optimal SSD Management with Predictions"** (rows 8–9) were added in a Sept 2026
  follow-up literature pass specifically covering 2025–2026 publications; they motivated Phase 8
  (`HYBRID_ROBUST_SMARTGC`) rather than the original Phase 0–7 restructuring.

Sources:
- <https://ieeexplore.ieee.org/document/11073159/> (Shiro)
- <https://www.sciencedirect.com/science/article/abs/pii/S0950705125020519> (iGC)
- <https://ieeexplore.ieee.org/document/11298441/> (AGC)
- <https://dl.acm.org/doi/10.14778/3717755.3717778> · <https://arxiv.org/abs/2406.01250> (DumpKV)
- <https://dl.acm.org/doi/full/10.1145/3725219> (SUP-GC)
- <https://dl.acm.org/doi/10.1145/3476886.3477520> (MiDA)
- <https://dl.acm.org/doi/10.1145/3816041> (ACM CSUR GC survey)
- <https://ieeexplore.ieee.org/document/11310889/> (NatSep)
- <https://dl.acm.org/doi/10.1145/3727122> (Optimal SSD Management with Predictions)
