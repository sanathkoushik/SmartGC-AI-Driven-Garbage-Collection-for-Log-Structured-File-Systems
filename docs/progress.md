# SmartGC Progress & Milestones

## Phase Status Summary

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton | **Completed** (`ca7a91d`) |
| **Phase 1** | Baseline LFS simulator + Greedy GC (C++17), unit tests, synthetic workload | **Completed** (`eb92fe2`) |
| **Phase 1b** | Foundation audit: GC reserve, admission control, config loading, contracts | **Completed** (`9e4049d`) |
| **Phase 2** | Real dataset acquisition, provenance, preprocessing | **Completed** (`15d3b90`) |
| **Phase 3** | Rewrite intervals, leakage-safe chronological sequences | **Completed** (`15d3b90`) |
| **Phase 4** | LSTM pretraining, hyperparameter search, baselines | **Completed** |
| **Phase 5** | Target fine-tuning, transfer comparison, prediction export | **Completed** (`a99d916`) |
| **Phase 6** | Simulator integration, multi-policy WAF experiments | **Completed** (`2cac459`) |
| **Phase 7** | Ablation, plots, analysis, conference demo | **Completed** |

---

## Phase 1b — Foundation Audit and Fixes

The committed Phase-1 engine reproduced its documented baseline **exactly** (WAF 1.0544, 272 blocks migrated, 53 GC invocations) and all 5 original tests passed. Six genuine defects were nevertheless found and fixed.

**1. Cleaning could exhaust the free pool at high utilization.** The old engine aborted at ≥95% live capacity — precisely the regime where hot/cold separation matters. Fixed with `gc_reserved_segments` (≥1), a pool only the collector may consume, plus a dedicated GC append point.

**2. Fully valid segments could be selected as GC victims** — migrating a whole segment and freeing nothing. Excluding them also gives the cleaning loop a termination proof: every pass strictly increases the free block count.

**3. No admission control.** Exceeding physical capacity surfaced as an obscure failure inside the collector; now `DeviceFullError` names the live count and the limit.

**4. `write()` ignored the request size**, accounting any request as one block.

**5. `config.yaml` was never read** and the metrics row hardcoded `"MIXED"`.

**6. A shared GC stream stranded user segments** (found while validating this phase). The write path opened a second segment instead of adopting the one cleaning had opened, leaving the first flagged open forever and excluded from victim selection — device full after ~3,500 writes at 24% utilization.

### Consequence for the Phase-1 baseline

| Engine | Migrated | GC | WAF |
| :--- | ---: | ---: | ---: |
| Phase 1 as committed | 272 | 53 | 1.0544 |
| Phase 1b, shared GC stream | 266 | 54 | 1.0532 |
| Phase 1b, separate GC stream (default) | 236 | 53 | **1.0472** |

Old and new shared-stream figures agree to within 0.1%; the residual is the reserve holding two segments back. Separating the GC stream cuts migrations 13% on its own. Utilization sweep in `results/metrics/foundation_validation.csv`.

---

## Phase 2 — Dataset Acquisition and Preprocessing

### Source decision

FIU, the requested secondary dataset, is **not obtainable** without submitting personal details to a web form. Four routes were probed directly on 2026-09-09: SNIA IOTTA returns HTTP 307 to a form requiring name, e-mail and organisation; the Harvey Mudd mirror runs the same gated application; the original FIU host times out; every ASU VISA Lab download link returns HTTP 404.

Selected instead, both anonymous over HTTPS:

| Role | Dataset | Source |
| :--- | :--- | :--- |
| Primary | MSR Cambridge (2007) | `cache-datasets` public S3 bucket |
| External generalization | SYSTOR '17 (2016) | the same bucket |

The bucket holds the **original distribution archives unmodified** — Microsoft's own `README.txt`, `DISCLAIMER.txt` and `MD5.txt` are inside them — so all 12 downloaded volumes were verified against a checksum manifest **written by the data's authors**. Nine candidate sources are scored in `results/dataset_source_comparison.csv`; the full record is `docs/dataset_source_decision.md`.

### Measured inventory (18 workloads, 3,000,000 raw records each)

| | |
| :--- | :--- |
| MSR volumes | 12, all eligible |
| SYSTOR LUNs | 6, of which 4 eligible |
| MSR write blocks | 1.93 M (`wdev_0`) to 25.0 M (`proj_0`) |
| MSR rewrite ratios | 0.823 to 0.983 |
| SYSTOR rewrite ratios | 0.363 to 0.569 |
| Excluded | `systor_lun2` (3,315 s < 3,600 s), `systor_lun6` (17,088 writes, 82 s) |

MSR rewrite ratios above 0.8 are the key measurement: these workloads overwhelmingly rewrite blocks they have written before, which is the regime where rewrite-interval prediction is meaningful at all.

### Roles, assigned by the frozen rule *before* any training

* **Pretraining (10):** `hm_0`, `mds_0`, `prn_0`, `proj_0`, `prxy_0`, `stg_0`, `ts_0`, `usr_0`, `wdev_0`, `web_0`
* **Held-out targets (2):** `rsrch_0`, `src2_0`
* **External (4 assigned, 1 run):** `systor_lun0`

### Verified transformations

Block expansion was checked against a genuine record: raw `128166372011600556,mds,0,Write,3201662976,20480,45361` (20 KB) expands to exactly blocks 781656–781660, all at timestamp 78997. A 500,000-record prefix of `mds_0` produced 930,507 block records from 136,949 multi-block and 27,269 unaligned requests, with **zero** records dropped.

---

## Phase 4 — Model

### Hyperparameter search (validation MAE only, 12 configurations)

Sequence length dominates; hidden size and learning rate barely matter.

| Sequence length | Best validation MAE (µs) |
| ---: | ---: |
| 5 | 8.16 × 10⁷ |
| 10 | 3.86 × 10⁷ |
| 20 | **2.56 × 10⁷** |

Selected: `sequence_length` 20, `hidden_dim` 64, 2 layers, dropout 0.1, lr 5e-4.

**Honest caveat:** the search capped each configuration at 5 epochs for tractability, and every configuration's best epoch was 4 or 5 — the models were still improving when it stopped. The search ranks configurations under a fixed budget; it does not establish converged performance.

### Pretrained model

| | |
| :--- | :--- |
| Traces | 10 MSR workloads (both targets excluded) |
| Samples | 1,050,011 train / 225,089 val / 224,900 test |
| Parameters | 50,497 |
| Epochs | 25 run, best at 22, no early stop |
| Training time | 108.4 min, CPU only |
| Validation MAE | 88,038,992 µs (log1p 0.9846) |
| Hot/cold threshold | 31,405 µs (training-set p30) |

---

## Phase 5 — Transfer Learning: the Four-Way Comparison

All four approaches scored on the **same** held-out chronological test split per target (final 15% of the trace). Full table with provenance: `results/ml/transfer_learning_comparison.csv`.

| Target | Approach | MAE (µs) | RMSE (µs) | MAE log1p | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `msr_rsrch_0` | best baseline (`median_interval`) | 127,907,767 | 995,424,552 | 4.2794 | 0.2348 |
| | LSTM from scratch | 121,085,055 | 952,439,480 | 1.0748 | 0.8800 |
| | Pretrained, no fine-tuning | 70,564,234 | 775,308,641 | 1.0069 | 0.8473 |
| | **Pretrained + fine-tuned** | **70,487,142** | **770,871,763** | **0.9507** | **0.8936** |
| `msr_src2_0` | best baseline (`median_interval`) | 149,625,578 | 737,595,174 | 4.9126 | 0.2017 |
| | LSTM from scratch | 162,729,978 | 759,434,109 | 1.1204 | 0.8734 |
| | Pretrained, no fine-tuning | 94,292,772 | 592,029,702 | 1.2082 | 0.8267 |
| | **Pretrained + fine-tuned** | **91,220,867** | **586,625,236** | **0.9735** | **0.8952** |
| `systor_lun0` | best baseline (`median_interval`) | 14,632,278 | 43,509,207 | 1.9662 | 0.5854 |
| | LSTM from scratch | 13,140,359 | 38,769,274 | 1.1870 | 0.7933 |
| | Pretrained, no fine-tuning | 13,615,229 | 38,874,552 | 1.3614 | 0.7356 |
| | **Pretrained + fine-tuned** | **11,436,713** | **33,754,339** | **1.1072** | **0.8023** |

**Prediction-level findings (all measured):**

1. **The LSTM beats every baseline decisively** — roughly 4× lower log-space error and 4× higher F1 on the MSR targets. Rewrite intervals *are* predictable, and a sequence model captures far more than a threshold on recent history.
2. **Pretraining alone does not reliably transfer.** It improves log-space error on `rsrch_0` (−0.068) but *degrades* it on `src2_0` (+0.088) and on `systor_lun0` (+0.174), even while cutting raw MAE ~40%.
3. **Fine-tuning always helps, and is the best model on every metric for all three targets.** It recovers and exceeds the scratch model in every case (−0.056, −0.235, −0.254 log1p vs pretrained).
4. The `last_interval` (persistence) baseline is the **worst** predictor of all (log1p 8.15–9.40); `median_interval` is the strongest baseline.

---

## Phase 6/7 — Does Better Prediction Reduce Write Amplification?

45 simulator runs: 3 workloads × 3 utilizations × 5 configurations, each replaying **1,000,000 identical write requests** against identical geometry and seed. Full table: `results/metrics/ablation.csv`.

### Mean change in WAF relative to MIXED

| Configuration | Mean | Best | Worst |
| :--- | ---: | ---: | ---: |
| **LSTM pretrained** | **−5.02%** | −13.12% | +0.07% |
| RULE_BASED | −4.10% | −12.83% | +0.15% |
| LSTM pretrained+finetuned | −3.53% | −12.07% | +0.12% |
| LSTM scratch | −3.07% | −8.27% | −0.03% |

### By utilization

| Configuration | 70% | 80% | 85% |
| :--- | ---: | ---: | ---: |
| LSTM pretrained | −7.7% | −5.1% | −2.4% |
| RULE_BASED | −7.0% | −3.9% | −1.4% |
| LSTM pretrained+finetuned | −6.5% | −3.0% | −1.0% |
| LSTM scratch | −5.8% | −2.5% | −1.8% |

### Storage-level findings — including the negative ones

1. **The chain breaks between prediction and placement.** `pretrained_finetuned` is the *best* predictor on all three workloads (Phase 5) but only the *third best* placement policy (−3.53% mean, against −5.02% for the un-adapted pretrained model). Better rewrite-interval prediction did **not** produce better hot/cold placement here.
2. **SmartGC's margin over the simple heuristic is modest**: −5.02% vs −4.10% mean, about 0.9 percentage points. The LSTM wins, but a running mean and one comparison gets most of the way there.
3. **Gains shrink sharply as utilization rises** — the opposite of the intuition that separation matters more under pressure. At 85% every configuration is within 2.4% of the baseline. With little free space, the collector must clean nearly-full segments regardless of how they were separated, so segregation has little room to help.
4. **External generalization is weak.** On `systor_lun0` no configuration beats MIXED by more than 1.9%, and at 85% utilization several are marginally *worse*. The MSR-pretrained model does not carry a useful placement advantage into a different workload family.
5. Prediction coverage is partial and measured, not assumed: 49–71% on MSR, 29–34% on SYSTOR. An LBA needs 20 rewrite intervals before it can be scored.

**Overall:** rewrite intervals are strongly predictable and the LSTM predicts them far better than any baseline, but that advantage translates into only a modest and utilization-dependent WAF reduction, and the best-predicting model is not the best-placing model.

---

## Defects Found During Validation

Beyond the six Phase-1b engine defects, one methodological defect was found and fixed in this project's own comparison.

**The rule-based control was thresholded with the wrong statistic.** The heuristic classifies on a *running mean* of an LBA's observed intervals, but it was being given the model's cutoff — a percentile of *next-interval targets*. Those distributions differ by orders of magnitude, because a running mean over an LBA's history is dominated by its long idle gaps: on `msr_rsrch_0` the model's cutoff is 69 µs while the p30 of the rule's own statistic is 16,330,010 µs, a factor of 236,000.

The effect was severe and silent: `RULE_BASED` labelled **2 writes HOT out of 550,987 classified**, so it was not separating by temperature at all — it was separating by *how much history a block had*, which correlates with being frequently rewritten. That produced a flattering but meaningless advantage for the heuristic (−27.5% at 85% utilization on `msr_rsrch_0`, against −5.2% for the best LSTM).

With the corrected threshold the heuristic labels ~28% of classified writes HOT, matching its intended p30, and the ranking reverses in several groups. `rule_threshold_from_training()` is now the single shared implementation, and two regression tests cover it.

**This is recorded because it changes the reported conclusion.** The uncorrected numbers would have supported "the simple heuristic beats the LSTM everywhere"; the corrected ones support "the LSTM wins modestly, and fine-tuning does not help placement".

---

## Known Limitations

* **Trace age.** MSR Cambridge 2007, SYSTOR '17 2016. SNIA files both as historical.
* **Truncation.** 3,000,000 raw records per trace, 150,000 sequences per trace, 1,000,000 writes per simulation — all chronological prefixes, all recorded in `results/final/experiment_manifest.json`.
* **External validation rests on one workload.** Four SYSTOR LUNs were assigned by the frozen rule; only `systor_lun0` was run. Cause: the machine has 15.6 GB RAM and free memory fell to 2.6 GB during training, cutting effective parallelism from ~13 threads to 3.8 and stretching one fine-tune from 17 minutes to over two hours. This is a compute-budget limitation, not a result-driven selection — `lun1`, `lun3` and `lun4` remain assigned in `results/dataset_roles.json` and the restriction is recorded in the manifest.
* **Single seed.** Replaying a real trace is deterministic, so repeated simulator runs are bit-identical and error bars over simulator seeds would be meaningless. The genuine variance source is model training; multiple training seeds were **not** run, so no variance is reported and none is implied.
* **No significance testing** was performed, and none is claimed.
* **The hyperparameter search was budget-capped** at 5 epochs; see Phase 4.
* **One device model** — a single greedy-cleaned log-structured device, fixed block and segment sizes, no parallelism, wear levelling or read-latency modelling.

---

## Decisions Log

1. **Decoupled architecture**: C++ and Python interact only through CSV contract files.
2. **Fixed random seeds**, explicit and logged.
3. **Strict novelty boundary**: hot/cold separation is an established concept; the scope is whether a learned interval predictor beats a threshold heuristic end to end.
4. **WAF = physical / logical**; GC migrations contribute to physical writes only.
5. **Over-provisioning is explicit** (Phase 1b): the collector owns a reserve, and usable capacity is derived from it.
6. **GC gets its own append point** (Phase 1b), retained as a configurable ablation.
7. **Real data only for results**: synthetic workloads are for tests and development.
8. **Workload selection frozen before training** (Phase 2), by a rule that consults no measured result.
9. **The rule-based control is thresholded on its own statistic** (Phase 6), so the comparison measures prediction quality rather than an artefact of unit mismatch.
