# Companion Study: Baseline Ladder, Learning-Augmented Placement, and Seed Robustness

**This branch is a companion to `main`, not a competing version of the project.**
`main` (the transfer-learning study: pretrain → fine-tune LSTMs on real MSR
Cambridge / SYSTOR '17 traces, with a leakage-audited pipeline) and this branch
were developed independently and in parallel by the two authors of this
project. They ask two different, complementary questions about the same
underlying problem — ML-driven hot/cold data placement to reduce write
amplification in a log-structured file system — and are intended to be
submitted together as two halves of one piece of work.

## How the two studies fit together

`main`'s headline finding is that **the chain breaks between prediction and
placement**: the pretrained + fine-tuned LSTM is the best *predictor* of
rewrite intervals on every one of its three real workloads, but only the
*fourth-best* placement policy once run through the deterministic simulator.
Better prediction did not reliably produce better hot/cold separation.

This branch investigates the natural follow-up question: **if prediction
quality doesn't reliably translate into placement quality, can the *way a
prediction is used* at placement time be made more robust to that gap** —
without retraining, and without assuming the prediction is trustworthy?

## What this branch contributes

1. **A falsifiable 6-rung baseline ladder** (`MIXED → RULE_BASED → SUP_LIKE →
   STAT_ML → LSTM_SMARTGC → LSTM_ATTN_SMARTGC`), so any "deep learning helps"
   claim is tested against zero-training and cheap-learning baselines rather
   than assumed. On a real OLTP trace (UMass SPC Financial1), the ladder
   reproduces `main`'s core finding from a different angle: the zero-training
   `RULE_BASED` heuristic (WAF 1.1879) beats every learned rung, including the
   full LSTM+attention model (WAF 1.2426) — the same "prediction doesn't
   guarantee placement gains" result, independently arrived at.

2. **`HYBRID_ROBUST_SMARTGC`** (`ml/inference/robust_blend.py`): a direct
   answer to the question above. It reuses the trained LSTM+attention
   checkpoint verbatim — no extra training — but blends its prediction with
   the `RULE_BASED` interval continuously, weighted by MC-dropout confidence
   and capped by a tunable `robustness_lambda`, instantiating the
   *consistency-robustness* tradeoff from Lange, Naor & Yadgar, **"Optimal SSD
   Management with Predictions"** (ACM SIGMETRICS 2025) — a paper not cited in
   `main`. Result: it recovers **~56%** of the WAF gap between the worst
   learned rung and the best baseline on the real trace (WAF 1.2123 vs. 1.2426
   / 1.1879), without any additional training cost.

3. **A 5-seed robustness check** (`experiments/seed_robustness.py`): every
   number in this branch (and, as far as we can tell, in `main`'s ablation
   table too) was originally reported from a single fixed seed. Rerunning the
   full pipeline across 5 seeds found that the ladder's overall shape is
   stable, but at least one close head-to-head comparison that looked solid at
   one seed (`HYBRID_ROBUST_SMARTGC` vs. `RULE_BASED`) only holds at 4 of 5
   seeds, and one rung (`LSTM_SMARTGC`) has 4–5x the seed-to-seed variance of
   every other rung. **We'd suggest running an equivalent seed check on
   `main`'s ablation numbers before final submission**, since the same
   single-seed caveat likely applies there.

## Headline results

**Real UMass SPC Financial1 trace, fixed GC trigger:**

| Policy | WAF |
|---|---|
| MIXED | 1.3015 |
| **RULE_BASED** (zero-training) | **1.1879** ← best |
| SUP_LIKE | 1.1919 |
| STAT_ML | 1.2213 |
| LSTM_SMARTGC | 1.2184 |
| LSTM_ATTN_SMARTGC | 1.2426 |
| HYBRID_ROBUST_SMARTGC (λ=0.65) | 1.2123 |

**Synthetic Zipf, 5-seed mean ± std:**

| Policy | mean WAF | std |
|---|---|---|
| LSTM_ATTN_SMARTGC | 1.0193 | 0.0124 (best at every seed individually) |
| HYBRID_ROBUST_SMARTGC | 1.0409 | 0.0048 |
| RULE_BASED | 1.0468 | 0.0029 |
| LSTM_SMARTGC | 1.0465 | **0.0160** (highest variance of any rung) |

Full detail, methodology, and every other finding: `docs/progress.md` (Phase
4a–9), `docs/related_work.md` (citations, including the two not in `main`:
NatSep, IEEE ICCD 2025; Lange/Naor/Yadgar, ACM SIGMETRICS 2025),
`docs/architecture.md` (implementation contracts).

## Reproducing

```bash
cmake -S simulator -B simulator/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build simulator/build --config Release
pip install -r requirements.txt
python -m experiments.run_matrix --op-sweep
python -m experiments.robustness_sweep
python -m experiments.seed_robustness
python -m ml.evaluation.plots
pytest -q
```

## Suggested framing for the joint submission

Rather than merging the two codebases (they diverge in simulator internals,
config schema, and the prediction file contract — a code-level merge is
substantial work neither of us should attempt under deadline pressure), we'd
suggest structuring the write-up as:

1. Intro / related work (merged citation list from both branches)
2. Dataset + methodology (from `main` — the more rigorous, leakage-audited
   real-trace pipeline)
3. Prediction quality: does transfer learning help? (`main`'s four-way
   comparison)
4. From prediction to placement: does it matter? (`main`'s ablation — "the
   chain breaks")
5. **Robustifying placement under an unreliable prediction signal** (this
   branch — the ladder + `HYBRID_ROBUST_SMARTGC`)
6. **How much of this survives across random seeds?** (this branch — Phase 9)
7. Limitations, future work

This keeps both authors' work fully visible and attributed, avoids any risk of
breaking `main`'s tested, working pipeline this close to a deadline, and reads
as one coherent research arc rather than two separate projects.
