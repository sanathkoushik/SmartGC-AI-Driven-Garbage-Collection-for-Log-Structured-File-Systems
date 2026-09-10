# SmartGC: presenting and demonstrating

How to run the demo, what to say, and how to answer the questions an evaluator
will actually ask.

**Every number in the talk must be read from `results/`.** This document
deliberately contains no result values: they change whenever the pipeline is
re-run, and a stale figure on a slide is worse than no figure. The demo prints
the live numbers; the tables below say which file each one comes from.

---

## Running the demo

**On stage, use the fast one.** It reads only the results a completed run
already wrote - it trains nothing and simulates nothing, so it finishes in about
a second and has nothing to fail:

```powershell
pwsh scripts/conference_demo.ps1
pwsh scripts/conference_demo.ps1 -Target msr_src2_0 -Utilization 85
```

The live version re-runs all three placement policies in front of the audience,
which is more convincing but takes about a minute and needs the trace present:

```bash
python scripts/conference_demo.py
```

Roughly a minute. It checks what exists, replays one real held-out workload
through all three placement policies, and prints the measured comparison. If a
prerequisite is missing it names the command that produces it rather than
inventing a number.

```bash
python scripts/conference_demo.py --trace msr_src2_0 --utilization 0.85
python scripts/conference_demo.py --max-writes 2000000
```

**Before the session:** run `python experiments/run_all.py` once so every
artefact exists, then run the demo once to warm the file cache and confirm the
output. Have `results/plots/` open in an image viewer.

---

## The 2-minute version

> Log-structured storage never overwrites in place. It appends, and later a
> garbage collector picks a segment, copies whatever is still live out of it, and
> erases it. Those copies are writes nobody asked for — that is write
> amplification, and it costs both endurance and bandwidth.
>
> The copies come from mixing. If a segment holds blocks that die in
> milliseconds next to blocks that live for hours, the long-lived ones get copied
> forward again and again. So: separate them at write time. Put data you expect
> to be rewritten soon in one log and data you expect to survive in another.
>
> That needs a prediction. We train a small LSTM on real block traces — Microsoft
> Research Cambridge, 2007 — to predict how long until a given block is rewritten
> next, from the history of its own previous rewrite intervals. We pretrain on ten
> workloads, hold two out entirely, and fine-tune on the early part of a held-out
> one.
>
> Then we feed those predictions into a deterministic LFS simulator and measure
> write amplification directly, against two controls: no separation at all, and a
> plain threshold heuristic on the same statistic. Same trace, same device, same
> seed — only the placement decision differs.
>
> [run the demo, read the table]

Then state the actual finding from `results/metrics/ablation.csv`. If SmartGC
wins, say by how much and at which utilization. If the heuristic wins, say that.
Both are results.

---

## The 5-minute version

Add, in this order:

**1. Why the prediction target is the rewrite interval.**
For each LBA, the gap between consecutive writes to it. The first write of an LBA
has no interval and is never used as a target — filling in a zero there would
teach the model that every newly written block is maximally hot. Reads never
produce intervals.

**2. Why transfer learning is the interesting part.**
Training a model per workload is impractical for a real device: it needs history
it does not have yet. So we ask whether a model pretrained elsewhere transfers.
Three variants, evaluated on the *same* held-out test split:

* `scratch` — trained on the target alone
* `pretrained` — the general model, applied unchanged
* `pretrained_finetuned` — the general model, adapted on the target's early data

`results/plots/14_transfer_learning_comparison.png` answers "does pretraining
help?" and "does fine-tuning help?" separately.

**3. Why the comparison is fair.**
Every policy replays the identical trace against identical geometry, working set,
GC parameters and seed. The heuristic uses the **same** hot/cold threshold as the
model and the **same** minimum history, so the two differ in prediction *quality*
and not in prediction *coverage*. The demo prints the write count and logical
bytes for every policy; they are identical by construction, and a test asserts it.

**4. The chain, and where it can break.**
```
better prediction -> better hot/cold classification -> fewer valid-block
migrations -> lower WAF
```
Each arrow is measured separately, which is the point: a link can fail. Better
MAE with no WAF change is a real and reportable outcome, and the plots are laid
out to show exactly which arrow broke.

**5. What we did not do.**
Old traces. One device model. Chronological prefixes rather than whole weeks.
FIU absent. See Limitations below — say these before you are asked.

---

## Where each number comes from

| Claim | File |
| :--- | :--- |
| Workload sizes, rewrite ratios, eligibility | `results/dataset_inventory.csv` |
| Which workloads were pretrained on / held out | `results/dataset_roles.json` |
| Hyperparameters and how they were chosen | `results/ml/hyperparameter_search.csv` |
| Prediction MAE / RMSE, model vs baselines | `results/ml/model_comparison.csv` |
| Hot/cold accuracy, precision, recall, F1, confusion | `results/ml/classification_metrics.csv` |
| WAF, GC bytes, migrations, per policy | `results/metrics/comparison.csv` |
| Change vs MIXED, the ablation | `results/metrics/ablation.csv` |
| Exact configuration, seed, versions, git commit | `results/final/experiment_manifest.json` |

---

## The figures, and what each one is for

| Figure | Shows |
| :--- | :--- |
| `01`–`02` dataset composition, write intensity | these are write-heavy workloads, so GC pressure is real |
| `03` rewrite-interval distribution | intervals span orders of magnitude — why the model works in log space |
| `04`–`05` LBA popularity, traffic concentration | the skew hot/cold separation exploits |
| `06` write rate over time | bursty; why a chronological split matters |
| `07` hot/cold class balance | what the classifier is up against |
| `08` training/validation loss | the model trains and early-stops honestly |
| `09`–`11` MAE, RMSE, log-space MAE vs baselines | **does the LSTM beat "just use the last interval"?** |
| `12` F1 and accuracy | prediction quality where it matters for placement |
| `13` confusion matrices | *how* it is wrong, not just how often |
| `14` scratch vs pretrained vs fine-tuned | **the transfer-learning result** |
| `15` WAF by policy | **the headline** |
| `16`–`17` migrations, GC bytes, GC count, physical bytes | the mechanism behind the WAF number |
| `18` WAF vs utilization | separation should matter more under pressure |
| `19` change vs MIXED | the summary slide |
| `20` prediction coverage | how many writes each policy could actually classify |

---

## Background an evaluator may probe

### What WAF is

`physical bytes written / logical bytes written`. The workload asks for the
logical bytes; the device writes those plus everything the collector copies while
reclaiming space. WAF 1.0 means no cleaning happened. WAF 3.0 means the device
wrote three bytes for every byte the workload asked for.

We measure it directly from counters in the simulator, and the identity
`physical = logical + gc_copied` is asserted after every run.

### Why hot/cold separation should help

Greedy cleaning picks the segment with the fewest live blocks. A segment of
purely short-lived data becomes almost entirely dead quickly and is nearly free
to reclaim. A segment of long-lived data stays live but is never chosen. A
*mixed* segment is the bad case: partly dead, so it gets picked, and the
long-lived remainder is copied forward — repeatedly.

### Hot/cold is not valid/invalid

This is the correctness question, and it is worth being crisp:

* **Valid/invalid** is a fact: does this block hold the newest version of its
  LBA? It decides what the collector migrates and when a segment can be erased.
* **Hot/cold** is a guess: when will this LBA be rewritten? It decides only which
  append point receives the write.

A block labelled COLD is invalidated by its next overwrite exactly like a HOT
one. A wrong temperature costs write amplification; it can never lose or corrupt
data. Two tests assert this directly.

### What the baselines mean

`MIXED` is the honest floor: a normal log-structured device with one append
point. `RULE_BASED` is the control that makes the ML claim falsifiable — a
running mean of an LBA's observed rewrite intervals against a threshold, holding
two numbers per LBA and doing one comparison. If the LSTM cannot beat that, the
sequence model is not earning its place, and that is the finding.

The heuristic is deliberately kept simple. Making it cleverer would turn it into
a hidden ML model and the comparison would stop meaning anything.

---

## Likely questions, with correct answers

**"Your traces are from 2007. Isn't that meaningless?"**
They are old, and we say so on the slide. They remain the standard public
block-I/O corpus for this kind of study, and SYSTOR '17 (2016 enterprise VDI) is
included precisely because it is a different era and workload family. The
mechanism under test — cleaning cost from mixing lifetimes in a segment — is a
property of log-structured storage, not of 2007 hardware. We do not claim these
workloads represent a modern SSD's traffic.

**"How do I know you didn't pick the workload that made this look good?"**
The eligibility thresholds and the rule that picks the held-out targets are in
`ml/dataset_selection.py`, they consult no measured result, and they were applied
by `experiments/analyze_datasets.py` before the first model was trained. The rule
is "sort candidates by trace id, take the two in the middle". Two tests assert
the assignment is order-independent and that a target never enters pretraining.

**"How do I know the test set didn't leak?"**
`tests/test_leakage.py` is the audit, and it runs. Splits are chronological with
boundaries advanced past tied timestamps; each trace is split before pooling; the
scaler and the hot/cold threshold are fitted on training data only;
hyperparameters are chosen on validation MAE; and `finetune.py` **refuses to run**
if a checkpoint's recorded pretraining traces contain its target. Some of those
tests inspect the artefacts an actual run produced, not a mock.

**"Isn't shuffling during training a leak?"**
No. Shuffling happens only *within* the training split, whose samples are all
earlier than every validation sample. What would leak is shuffling before the
split, which the pipeline never does.

**"Your model only predicts some of the writes."**
Correct, and the fraction is reported per run rather than assumed. An LBA needs
`sequence_length` rewrite intervals of history before it can be scored. Writes
without a prediction are placed on the main log and counted as
`unpredicted_writes`. We did not invent predictions to fill the gap, and the
heuristic is held to the same minimum history so coverage is matched.

**"Why not compare against F2FS / a real FTL?"**
Out of scope, and we would not be able to attribute the difference. The
comparison here is controlled: one engine, one workload, one variable. Hot/cold
separation is not claimed as novel — it is in F2FS and throughout the FTL
literature. The question is narrower: does a *learned* interval predictor beat a
threshold on the same statistic, end to end?

**"What if the improvement is just the GC stream separation?"**
It is a config flag (`gc_separate_stream`) and it is held constant across every
policy in a comparison, so it cannot be the explanation for a difference between
them. Its isolated effect was measured separately during foundation validation
and is recorded in `docs/progress.md`.

**"Would this work in a real device?"**
Unknown from this work, and we do not claim it. The inference cost, the memory
for per-LBA history, and the behaviour under workload shift are all unaddressed.
What we can say is whether the prediction quality is there and whether it
survives the trip to WAF in a controlled setting.

**"Why microseconds?"**
It is the finest unit all three source formats genuinely support — SYSTOR's own
README disclaims its nanosecond digits — so no source is given false precision.

---

## Limitations — state these unprompted

* **Trace age.** MSR Cambridge 2007, SYSTOR '17 2016. SNIA files both as
  historical.
* **Truncation.** Experiments use chronological prefixes of each trace, not whole
  weeks. The caps are recorded in the experiment manifest.
* **One device model.** A single greedy-cleaned log-structured device with fixed
  block and segment sizes. No parallelism, wear levelling, or read-latency
  modelling.
* **FIU is absent.** External generalization rests on SYSTOR '17 alone, because
  FIU could not be obtained without submitting personal details to a form.
* **Determinism cuts both ways.** Replaying a real trace is deterministic, so
  repeated simulator runs are bit-identical and error bars over simulator seeds
  would be meaningless. Real variance comes from training seeds; where multiple
  seeds were not run, that is stated rather than implied.
* **No significance testing** unless a test was actually run.
* **Prediction coverage is partial**, as above.

---

## If the demo fails on stage

Every failure path names the command that fixes it. In order of likelihood:

| Symptom | Fix |
| :--- | :--- |
| Anything at all goes wrong live | `pwsh scripts/conference_demo.ps1` - reads committed results only |
| "results/dataset_roles.json is missing" | `python experiments/run_all.py --stages normalize inventory` |
| "has not been normalized" | `python -m ml.preprocessing.normalize --discover` |
| "no trained model for ..." | `python experiments/run_all.py` |
| "smartgc_sim not found" | `python scripts/build_simulator.py` |
| No raw data at all | `python scripts/download_datasets.py --dataset all` |

Fallback if the network or the machine is uncooperative: the committed CSVs under
`results/` and the figures in `results/plots/` are the same numbers the demo
prints. Present those and say the demo regenerates them.
