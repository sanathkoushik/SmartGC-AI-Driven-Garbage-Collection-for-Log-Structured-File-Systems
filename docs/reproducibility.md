# Reproducing SmartGC

Every number in this repository is produced by the commands below. Nothing is
transcribed by hand.

---

## 0. Prerequisites

| | Requirement |
| :--- | :--- |
| Compiler | C++17 (GCC 9+, Clang 10+, MSVC 2019+) |
| Build | CMake 3.15+ |
| Python | 3.10+ |
| Disk | ~1 GB for raw traces, ~4 GB for normalized intermediates |
| Network | anonymous HTTPS; no account or credentials |
| GPU | not required — the model is small and trains on CPU |

```bash
# Windows
winget install Kitware.CMake
winget install Python.Python.3.12

# Debian / Ubuntu
sudo apt install cmake build-essential python3-venv
```

---

## 1. Environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

Use `.venv/bin/python` on Linux and macOS. The CPU-only PyTorch index is worth
using: the default wheel pulls a multi-gigabyte CUDA runtime this project never
touches.

---

## 2. Build the simulator

```bash
python scripts/build_simulator.py --test
```

This locates CMake (including a `winget` install not yet on `PATH`), picks a
working generator, builds, and runs the C++ suite. `simulator/CMakeLists.txt` is
authoritative; the plain commands work too:

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --config Release
./simulator/build/simulator_tests
```

> On MSYS2/MinGW the build links the GCC runtime statically. This is required,
> not cosmetic: some MSYS2 installations cannot link the shared runtime at all —
> `ld` exits with status 116 and no diagnostic for any code that uses exceptions.

---

## 3. Get the data

```bash
python scripts/download_datasets.py --dataset all
python scripts/verify_datasets.py
```

No form, no account. Individual volumes are pulled out of the published archives
by HTTP range request, so the ~5 GB of archives are never fetched in full.
Transfers resume after a dropped connection.

`verify_datasets.py` checks MSR volumes against `MD5.txt` **written by the data's
authors**, checks everything else against the download manifest, and parses the
first 2,000 records of each file so a truncated download cannot masquerade as a
workload property.

Useful variants:

```bash
python scripts/download_datasets.py --list                    # what is available
python scripts/download_datasets.py --dataset msr --volumes hm_0 prxy_0
python scripts/download_datasets.py --dataset msr --all-volumes   # all 36, ~5 GB
python scripts/download_datasets.py --dataset fiu             # prints manual steps
```

---

## 4. Run everything

```bash
python experiments/run_all.py
```

That runs all nine stages in order and writes
`results/final/experiment_manifest.json` recording the git commit, the full
configuration, the seed, the workload roles, the caps in force, and the
Python/PyTorch/platform versions.

A smoke test that exercises every stage in a few minutes — **not for reportable
results**, and labelled as such in the manifest:

```bash
python experiments/run_all.py --quick
```

Individual stages:

```bash
python experiments/run_all.py --stages normalize inventory
python experiments/run_all.py --stages tune pretrain models
python experiments/run_all.py --stages evaluate predict
python experiments/run_all.py --stages simulate analyze
```

---

## 5. The same pipeline, one stage at a time

Each stage is a normal module with its own `--help`; `run_all.py` only sequences
them.

**Preprocess** — raw traces to the normalized contract, plus per-workload statistics:

```bash
python -m ml.preprocessing.normalize --discover --max-records 3000000
python -m ml.preprocessing.normalize --dataset msr --input data/raw/msr/hm_0.csv.gz
```

**Inventory and workload selection** — measured statistics and the frozen role
assignment. Run this *before* training; that ordering is what makes
"we did not cherry-pick the workload" checkable:

```bash
python experiments/analyze_datasets.py --sources
```

Writes `results/dataset_inventory.csv`, `results/dataset_roles.json` and
`results/dataset_source_comparison.csv`.

**Hyperparameter search** — validation MAE only, test data untouched:

```bash
python -m ml.training.tune --traces $(...pretraining traces...) --epochs 8
```

Writes `results/ml/hyperparameter_search.csv` and
`hyperparameter_search_selected.json`.

**Pretrain** the general model on the pretraining workloads:

```bash
python -m ml.training.pretrain --traces msr_hm_0 msr_mds_0 msr_prn_0 ... \
    --output models/pretrained/msr_pretrained.pt
```

**Adapt** to each held-out target — the three-way comparison:

```bash
python -m ml.training.finetune --trace msr_prxy_0 --from-scratch
python -m ml.training.finetune --trace msr_prxy_0 --no-adapt \
    --checkpoint models/pretrained/msr_pretrained.pt
python -m ml.training.finetune --trace msr_prxy_0 \
    --checkpoint models/pretrained/msr_pretrained.pt
```

**Evaluate** models and baselines on the same held-out split:

```bash
python experiments/evaluate_models.py --targets msr_prxy_0 msr_proj_0
```

**Export predictions** for the simulator:

```bash
python -m ml.inference.predict --trace msr_prxy_0 \
    --checkpoint models/finetuned/pretrained_finetuned__msr_prxy_0.pt \
    --model-version pretrained_finetuned
```

**Simulate** — the three policies on one identical workload:

```bash
./simulator/build/smartgc_sim --config config/config.yaml \
    --trace data/processed/normalized/msr_prxy_0.csv \
    --total-segments 1400 --placement MIXED --model-type none \
    --export-metrics results/metrics/comparison.csv --quiet

./simulator/build/smartgc_sim --config config/config.yaml \
    --trace data/processed/normalized/msr_prxy_0.csv \
    --total-segments 1400 --placement RULE_BASED --model-type rule \
    --rule-threshold-us 12345.0 --rule-min-history 10 \
    --export-metrics results/metrics/comparison.csv --quiet

./simulator/build/smartgc_sim --config config/config.yaml \
    --trace data/processed/normalized/msr_prxy_0.csv \
    --total-segments 1400 --placement LSTM_SMARTGC \
    --model-type pretrained_finetuned \
    --predictions data/predictions/pretrained_finetuned__msr_prxy_0.csv \
    --export-metrics results/metrics/comparison.csv --quiet
```

`--total-segments` and `--rule-threshold-us` are computed by `run_all.py` from
the workload (see `size_device` and `_rule_threshold`); the values above are
placeholders for the shape of the command.

**Analyse** — tables and figures from the measured CSVs:

```bash
python experiments/analyze_results.py
```

---

## 6. Demonstration

Fast, reads only generated results (no training, no simulation):

```powershell
pwsh scripts/conference_demo.ps1
```

Live, re-runs the three placement policies:

```bash
python scripts/conference_demo.py
python scripts/conference_demo.py --trace msr_proj_0 --utilization 0.85
```

Replays one real workload through all three policies and prints the measured
comparison. If an artefact is missing it names the command that produces it
instead of printing a number.

---

## 7. Audit

```bash
python experiments/audit.py            # report
python experiments/audit.py --strict   # non-zero exit on any failure
```

`tests/test_leakage.py` checks the *code* obeys the methodology; this checks the
*artefacts a run produced* do. It verifies publisher checksums, that targets were
held out of pretraining, that every approach for a target shares one sequence
length and one test split, that every policy group replayed an identical
workload, the `physical == logical + gc_copied` identity, and that no policy
collapsed to an all-hot or all-cold labelling. Three defects in this project were
found this way, each of which had passed the unit tests.

---

## 8. Tests

```bash
python scripts/build_simulator.py --test     # C++ suite
python -m pytest tests/ -q                   # Python suite
python -m pytest tests/test_leakage.py -v    # the leakage audit on its own
```

Run the Python suite from the repository root. Three tests skip unless real
traces have been downloaded; the leakage tests that inspect trained checkpoints
skip until a run has produced them.

---

## 9. What determines a result

| Input | Where it is fixed |
| :--- | :--- |
| Random seed | `random_seed` in `config/config.yaml`, or `--seed` |
| Model hyperparameters | selected by `tune`, recorded in `hyperparameter_search_selected.json` |
| Splits | `train_split` / `val_split` / `test_split` in `config.yaml` |
| Hot/cold cutoff | `hot_percentile_cutoff`, applied to **training** targets |
| Device geometry | `simulator:` section, plus `size_device()` from the workload |
| Caps | `--max-records`, `--max-sequences`, `--max-writes` (recorded in the manifest) |

All of it is captured in `results/final/experiment_manifest.json` alongside the
git commit, so any table in `results/` can be traced back to the exact
configuration that produced it.

**Determinism.** Replaying a real trace is deterministic: repeated simulator runs
on the same trace, geometry and predictions are bit-identical, so error bars over
simulator seeds would be meaningless. The genuine source of variance is model
training. Where several training seeds were run, mean and standard deviation are
reported; where they were not, that is stated rather than implied.
