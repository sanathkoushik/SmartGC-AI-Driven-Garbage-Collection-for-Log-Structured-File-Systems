# SmartGC methodology

What was decided in advance, what the pipeline is allowed to look at when, and
how each rule is checked. Every claim here corresponds to code that runs and, in
most cases, to a test that fails if the rule is broken.

---

## 1. The question

> Does a lightweight LSTM trained on real block-I/O rewrite history predict the
> next rewrite interval well enough — after pretraining on several workloads and
> fine-tuning on a held-out one — to place blocks better than a simple threshold
> heuristic, and does that better placement actually reduce write amplification?

Five configurations are compared on identical workloads:

| # | Configuration | Placement | Prediction source |
| :-- | :--- | :--- | :--- |
| 1 | `MIXED` | one append point | none |
| 2 | `RULE_BASED` | hot/cold | running mean of an LBA's observed rewrite intervals |
| 3 | `LSTM_SMARTGC` / `scratch` | hot/cold | LSTM trained only on the target workload |
| 4 | `LSTM_SMARTGC` / `pretrained` | hot/cold | LSTM pretrained on other workloads, unadapted |
| 5 | `LSTM_SMARTGC` / `pretrained_finetuned` | hot/cold | pretrained, then adapted on the target's early data |

3 vs 4 vs 5 is the transfer-learning ablation; 2 vs {3,4,5} is the "does ML earn
its place" comparison; 1 is the baseline everything is measured against.

**A negative result is a result.** If the heuristic wins, or prediction improves
without WAF improving, that is what gets reported.

---

## 2. Workload selection — frozen before any model was trained

Implemented in `ml/dataset_selection.py`, applied by
`experiments/analyze_datasets.py`, recorded in `results/dataset_roles.json`.

### Eligibility

A workload is a candidate only if **all** of these hold:

| Criterion | Threshold | Why |
| :--- | ---: | :--- |
| `write_blocks` | ≥ 200,000 | enough write traffic to fill the simulated device several times and actually trigger cleaning |
| `rewrite_ratio` | ≥ 0.30 | a workload that writes each block once has no rewrite intervals; the question is undefined on it |
| `unique_written_lbas` | ≥ 5,000 | a working set larger than a handful of segments, so placement is not trivial |
| `duration_seconds` | ≥ 3,600 | long enough that a chronological 70/15/15 split leaves a meaningful test period |

### Role assignment

Candidates are sorted by `trace_id` — a stable key that depends on the file name,
not on any measurement. The **two MSR candidates at the middle of that
alphabetical order** become the held-out targets; every other MSR candidate
becomes a pretraining workload. SYSTOR candidates are never used for pretraining
or target selection; they are the external-generalization set.

**This rule cannot be gamed by looking at results**, because it consults no
result. `test_role_assignment_is_independent_of_input_order` asserts the
assignment does not depend on the order statistics are supplied in, and
`test_role_assignment_holds_targets_out_of_pretraining` asserts the target never
appears in the pretraining set.

`ml/training/finetune.py` additionally **refuses to run** if a checkpoint's
recorded pretraining traces contain the target it is being adapted to. Leakage of
that kind is an error, not a warning.

---

## 3. Preprocessing

### Normalization

Every dataset is reduced to one contract (`docs/architecture.md` §2.1):

```
timestamp,lba,size,operation,trace_id
```

* `timestamp` — **microseconds since that trace's first record**. Chosen so
  intervals are comparable across datasets whose native units differ (Windows
  FILETIME ticks vs nanoseconds vs fractional Unix seconds). Conversion happens
  per record and rebasing afterwards, so the reported interval between two
  records is the difference of their truncated microsecond values.
* `lba` — block index at the configured block size (4 KB).
* `size` — always 1. Multi-block requests are expanded.
* `operation` — `W`, `R` or `D`.

### Block expansion

A request covers **every block it touches**, including partial first and last
blocks:

```
first_block = offset // 4096
last_block  = (offset + size - 1) // 4096
```

A 20 KB aligned request becomes 5 records; an unaligned 8 KB request becomes 3.
This is verified against real MSR records in
`test_msr_parser_expands_and_converts`, and the simulator refuses any record
whose size is not exactly one block, so an unexpanded request can never be
silently accounted as 4 KB.

Unaligned requests are counted (`unaligned_requests`) rather than rounded away.

### Device-namespace safety

**LBAs are only meaningful within one volume.** Two disks both have a block 0;
pooling them would invent rewrites between unrelated data. Each parser reports a
device key per record — `<host>_<disk>` for MSR, `major:minor` for FIU, `LUN` for
SYSTOR — and a normalized trace always describes exactly one device. The default
policy **rejects** a file containing more than one, naming the devices found.

This is why the normalized schema carries no device column: it would be constant
within a file. The identity lives in `trace_id` and in `device_selected` in the
statistics JSON.

### Rejected records

Every dropped record is counted by category and reported: malformed, bad offset,
bad size, oversized request, unknown operation (with the offending spellings
recorded), and other-device. Nothing is discarded silently.

---

## 4. Rewrite intervals

For each LBA, the rewrite interval of a write is the time since the previous
write **to that same LBA**:

```
LBA 100 written at t = 10, 18, 24, 40
intervals             =  8,  6, 16
```

* **The first write of an LBA has no interval and never becomes a target.** A
  fabricated zero would teach the model that every newly written block is
  maximally hot. `test_first_write_never_produces_a_target` asserts this.
* A genuine zero (two writes in the same microsecond) is kept — it is a real
  observation.
* **Reads never produce rewrite intervals.** They may appear in a normalized
  trace and are ignored here.

A training sample is `L` consecutive intervals of one LBA predicting the next.
Windows never cross an LBA boundary. Each sample is stamped with the timestamp of
the write that produced its **target** — the moment the label becomes observable,
and the key the chronological split orders by.

---

## 5. Leakage prevention

### The rules

1. **Chronological splits only**, 70/15/15 by default, never random.
2. **Split each trace first, then pool the matching splits.** Pooling before
   splitting would put one workload's future into another's training data.
3. **A split boundary is advanced past tied timestamps**, so no instant appears
   on both sides and `max(train) < min(val) < min(test)` holds strictly.
4. **Normalization statistics come from training intervals only.**
5. **The hot/cold threshold comes from training targets only** — a percentile of
   them, never of test targets.
6. **Hyperparameters are selected on validation MAE only.**
7. **The target workload is excluded from pretraining.**
8. **Fine-tuning uses only the target's early training split**, never its later
   test region.
9. **Test data is read exactly once**, at final evaluation.
10. Shuffling happens **only within the training split** during batching. Those
    samples are already all earlier than every validation sample, so reordering
    them cannot leak; what would leak is shuffling before the split, which the
    pipeline never does.

### The audit

`tests/test_leakage.py` is the executable form of this checklist:

| Check | Test |
| :--- | :--- |
| No future interval enters an input window | `test_no_future_interval_enters_an_input_window` |
| Splits strictly ordered in time | `test_splits_are_strictly_ordered_in_time` |
| No sample in two splits | `test_no_sample_appears_in_two_splits` |
| Boundary never cuts a tied timestamp | `test_split_boundary_never_cuts_a_tied_timestamp` |
| Scaler fitted on train only | `test_scaler_depends_only_on_training_data` |
| Threshold from train only | `test_hot_threshold_depends_only_on_training_data` |
| Fine-tuning uses early target data only | `test_finetuning_uses_only_the_early_part_of_training` |
| Target held out of pretraining | `test_role_assignment_holds_targets_out_of_pretraining` |
| Trained checkpoints honour that | `test_trained_models_did_not_pretrain_on_their_target` |
| Every checkpoint records a train-derived threshold | `test_trained_models_record_a_training_derived_threshold` |
| Hyperparameters chosen on validation | `test_hyperparameters_were_selected_on_validation_only` |
| No random temporal shuffling | `test_chronological_split_rejects_unsorted_input` |

The last three inspect artefacts produced by a real run and skip when none exist,
so they check the actual experiment rather than a mock of it.

---

## 6. Controlled comparison in the simulator

### Identical workloads

Every policy replays the **same normalized trace, in the same order, against the
same device geometry, working set, GC parameters and seed**. Only the placement
decision differs. `test_policies_receive_identical_workload` asserts that the
request count and logical bytes written are identical across all three policies;
`test_end_to_end_all_three_policies` asserts the same through the built binary.

### Device sizing

The device is sized from the workload, not the other way round:

```
segments = ceil(distinct_written_lbas / (utilization * blocks_per_segment))
```

then increased until the working set fits the usable capacity of the **most
restrictive** policy (two user append points). Every policy in a comparison then
runs on that same geometry, so the segregated policies are not quietly given more
or less room than `MIXED`.

### Matched prediction coverage

The LSTM cannot score a write until its LBA has `L` rewrite intervals of history.
The heuristic could in principle classify after one. To keep the comparison about
prediction *quality* rather than prediction *coverage*, `--rule-min-history` is
set to the model's sequence length, so both stay silent on the same early writes.
Both also use the **same** training-derived threshold. Coverage is reported per
run (`unpredicted_writes`, `prediction_coverage`) rather than assumed.

### Fallback

A write with no prediction is placed on the main log and counted. It is never
guessed at, and a prediction file that is missing, malformed or carries an
unreadable class label produces a counted `UNKNOWN`, never a silent `COLD`. A
prediction-driven policy invoked without a prediction file is a hard error —
silently degrading to `MIXED` would invalidate the comparison.

### Hot/cold is not valid/invalid

Placement never affects validity. A block labelled `COLD` is invalidated by its
next overwrite exactly like a `HOT` one; a mispredicted temperature costs write
amplification and can never lose data.
`test_temperature_routes_placement_only` and
`test_replay_preserves_validity_semantics` assert this directly.

---

## 7. What limits the results

Recorded here so the write-up does not have to discover them later.

* **Trace age.** MSR Cambridge is 2007; SYSTOR '17 is 2016. Neither represents
  contemporary storage hardware or workloads.
* **Truncation.** Sequences per trace and writes per simulation run are capped
  (both recorded in `results/final/experiment_manifest.json`). Caps are always a
  **chronological prefix**, never a sample, so causal order survives — but the
  results describe that prefix, not the whole week.
* **Determinism removes one source of variance and not another.** Replaying a
  real trace is deterministic, so repeated simulator runs are bit-identical and
  error bars over simulator seeds would be meaningless. Genuine variance comes
  from model training seeds; where multiple seeds were run this is reported as
  mean and standard deviation, and where they were not, that is stated rather
  than implied.
* **No statistical significance is claimed** unless a test was actually run.
* **FIU is absent.** The external-generalization claim rests on SYSTOR '17 alone.
* **One device model.** A single greedy-cleaned log-structured device with a
  fixed block and segment size; no parallelism, no wear levelling, no read
  latency modelling.
