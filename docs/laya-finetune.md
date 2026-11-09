# Fine-tuning Laya on a GPU box

What to run, in order, on a machine with a CUDA GPU. Written for a 40 GB A100,
which other users can share; the notes say where the card or its load changes
something.

The deliverable is **not a Laya checkpoint**. It is the three-way comparison —
Laya vs the roberta-base fine-tune vs TF-IDF + logistic regression, on identical
splits through one eval harness — plus Laya's Hindi numbers, which neither of
the other two can produce. A model nobody compared to the baseline is not a
result.

---

## 0. Before you start: what is and isn't done

| | state |
|---|---|
| `modeling/text/laya_clf.py` | written, 24 tests in `tests/test_laya.py` (22 pass, 2 skip because laya is installed). The real checkpoints load through `LayaScorer` in a smoke test on 2026-11-09 with laya 0.4.30, **base checkpoints only**. No fine-tune has run |
| `configs/models.yaml → laya` | written: routing, question, train knobs, precision target |
| Training JSONL exporter | written (`LayaTrainingExporter`), tested |
| `data/benchmarks/` | `make benchmarks` supplies LIAR (zip, unattended), the CoAID repository clone, and the FakeNewsNet repository clone. The FakeNewsNet clone has the label CSVs and the crawler, not article text. Checksums go in `data/benchmarks/manifest.json` |
| The fine-tune | not run |
| Scoring integration | `modeling/scoring.py` has **no Laya stage**, and `LayaScorer.checkpoint_for` cannot yet select a fine-tuned directory (it returns only the configured base checkpoint). A finished fine-tune is **not yet reachable from scoring** |
| The eval harness for the three-way comparison | **not written** — see step 6 |

Everything up to step 5 is runnable now. Step 6 is the part that still needs
code, and it is the part that produces the result.

---

## 1. Environment

Run everything in `tmux`, so a dropped SSH session does not kill a run:

```bash
tmux new -s laya          # or: tmux attach -t laya
python3 --version         # 3.10+ required by laya; 3.11-3.13 is the safe range
nvidia-smi
```

GPU hosts often have a small root disk. Before any install or download, point
the caches at a directory on a large disk. Do this in **every** new shell and
tmux window. Without it, a checkpoint download can fill the root disk and fail
halfway.

```bash
SCRATCH=/path/to/large/disk
export HF_HOME=$SCRATCH/hf
export PIP_CACHE_DIR=$SCRATCH/pip
export UV_CACHE_DIR=$SCRATCH/uv
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"
```

```bash
git clone https://github.com/TEJAS-SAI-PRASHAD-K/narrative-intelligence.git
cd narrative-intelligence
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[modeling,laya,sources,dev]"
```

Sanity check that the optional dependency actually resolved:

```bash
.venv/bin/python -c "import laya, torch; print(laya.__version__, torch.__version__, torch.cuda.is_available())"
```

> The published docs say Laya needs `transformers` 5.x and `torch` 2.14. The
> **wheel's own metadata** declares `transformers>=4.48.0` and `torch>=2.0.0`,
> which this project's `>=4.40` / `>=2.2` already satisfy. Metadata is what pip
> resolves against, so there is no conflict and no pin was moved. If a resolver
> error mentions transformers, it is not this.

---

## 2. Benchmarks

```bash
make benchmarks
```

Fetches LIAR, CoAID and FakeNewsNet into `data/benchmarks/` with checksums. The
`sources` extra from step 1 provides what the fetcher needs.

**All three are required.** The `misinfo` classifier was trained on the union of
LIAR, FakeNewsNet and CoAID. A Laya model trained on LIAR alone sees a different
training set, so its numbers are not like for like with `misinfo` and the
comparison fails. LIAR alone is not a shortcut.

Two caveats carried over from the Phase 1 README, one still true as written and
one corrected:

- **FakeNewsNet ships ids and a crawler, not article text.** You do not need the
  crawl. The label CSVs in the clone (`dataset/politifact_*.csv`,
  `dataset/gossipcop_*.csv`) are enough for the loader that this project uses.
- **CoAID's tweet content cannot be recollected** in this project.

---

## 3. Why the training data is the benchmarks and not the Indian verdicts

`data/factchecks/` is **evaluation only**. Published fact-checks are
single-class — a measured 96 `false` and 0 `true` — because debunking is news
and confirming is not. Nothing can be trained on one class.

So: train on the union of LIAR (US political statements), FakeNewsNet and
CoAID, and evaluate on Indian recontextualised media. The domain gap is not a
flaw in this design, it is the quantity being measured. See the README section "Fact-check supervision".

---

## 4. Export the training JSONL

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from modeling.config import get_settings
from modeling.datasets import get_dataset
from modeling.datasets.splits import group_train_val_test
from modeling.text.laya_clf import LayaTrainingExporter
from modeling.text.misinfo_clf import build_training_frame

settings = get_settings()
OUT = Path("data/laya")

# The same union, the same frame builder and the same splitter call that
# `modeling/training.py` uses for misinfo. Same seed, same groups, same split.
loaded = {key: get_dataset(key).load().frame for key in ("liar", "fakenewsnet", "coaid")}
frame = build_training_frame(loaded)

# Group-aware split, NOT a random one. LIAR repeats speakers, so a random
# split leaks a speaker's style across the boundary and inflates every metric.
# `split.train`, `split.val` and `split.test` are positional indices into
# `work`, not into `frame`: always index `work`.
work, split = group_train_val_test(
    frame, group_col="group_id", label_col="label", seed=settings.seed
)
print(split.describe())

exporter = LayaTrainingExporter()   # reads the "text" column of the frame
for name, idx in (("train", split.train), ("val", split.val)):
    n = exporter.export(work.iloc[idx], OUT / f"{name}.jsonl", label_column="label")
    print(name, n)
# `split.test` is NOT exported. It is held for step 6 and must never reach finetune.
PY
```

On the public data this prints
`grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)`.
If your numbers differ, the benchmarks differ from the ones `misinfo` used, and
the comparison is no longer like for like.

> **Split on the group key, and use the same splitter `misinfo` used.** LIAR
> repeats speakers; a random split puts the same speaker on both sides and
> inflates every number. More importantly, a Laya figure computed on a
> *different* split than the roberta baseline is not comparable to it, and the
> comparison is the whole deliverable.

**Use this exporter, not a CSV.** `laya.train.read_data` sends a `.csv` to
`rows_from_csv`, which "discovers distinct labels in `label_column` ... and
builds a **choice** question". A CSV of `text,label` therefore trains a
two-option *choice* head, while `modeling/text/laya_clf.py` asks a **`noul`** at
inference. Training one head and querying another degrades quietly rather than
erroring, and the result looks like a weak model instead of a wiring mistake.

The exporter takes its question from `LayaScorer.question()` — the same config
the scorer reads — so train/inference drift cannot happen.

Check the first row before spending GPU hours on it:

```bash
head -c 600 data/laya/train.jsonl | python3 -m json.tool 2>/dev/null | head -20
```

`expected.misinfo` must be `true` for a false claim. If that is inverted,
everything downstream is inverted and ROC-AUC near 0 will read as a bad model.

---

## 5. Fine-tune both checkpoints

Fine-tune the English checkpoint and the multilingual checkpoint, each into its
own output directory. `finetune` resolves its `model_dir` argument through
`resolve_checkpoint_dir`, which accepts a local checkpoint directory, a Hub id,
or an alias. The aliases `"english"` and `"multilingual"` download only that
checkpoint's files from `convaiinnovations/laya`, into the cache that
`HF_HOME` sets. No separate download step is needed.

Held-out validation data goes in `TrainConfig(eval_data=...)`, which takes a
string path to a JSONL file in the same format as the training file. Without
it, `finetune` evaluates on its own calibration slice and labels the result
`in_sample_calibration`, and that number must not be reported as validation.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from laya.train import TrainConfig, finetune
from modeling.config import module_config

cfg = module_config("laya")["train"]
runs = (
    ("english", "models/laya/en-v0.1.0"),
    ("multilingual", "models/laya/hi-v0.1.0"),
)
for alias, out_dir in runs:
    summary = finetune(
        "data/laya/train.jsonl",
        alias,
        out_dir,
        TrainConfig(
            epochs=cfg["epochs"],
            micro_batch=cfg["micro_batch"],
            grad_accum=cfg["grad_accum"],
            encoder_lr=cfg["encoder_lr"],
            head_lr=cfg["head_lr"],
            weight_decay=cfg["weight_decay"],
            grad_clip=cfg["grad_clip"],
            loss=cfg["loss"],
            rl_samples=cfg["rl_samples"],
            calib_frac=cfg["calib_frac"],
            calib_max=cfg["calib_max"],
            calib_seed=cfg["calib_seed"],
            eval_data="data/laya/val.jsonl",
        ),
    )
    print(alias, summary)
    Path(out_dir, "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
PY
```

Run it inside `tmux`. The two runs go one after the other, and each writes to its
own directory.

Notes that matter:

- **Fine-tuning is mandatory.** Vendor-reported, the base checkpoint scores
  0.362 on typed-decisions against a 0.461 majority-class baseline — *worse than
  guessing*. Run the zero-shot number too, but report it as a baseline.
- **The training data is English.** LIAR, FakeNewsNet and CoAID are all English
  text. The multilingual fine-tune therefore does **not by itself prove any
  transfer to Hindi**. Whether it transfers is an open question that step 6
  measures. Also report the **base multilingual checkpoint, with no
  fine-tune, as a baseline on Hindi**. If the fine-tuned model does not beat that
  baseline on Hindi, the fine-tune added nothing there.
- **Calibration happens inside `finetune`.** It holds out `calib_frac` (0.1) to
  fit one temperature per (question type, option count). Vendor-reported mean
  ECE 0.466 → 0.081. This is why the module has no separate `Calibrator`, unlike
  `misinfo`.
- The defaults assume the published notebook's effective batch of 64
  (8 × 2 GPUs × 4 accumulation). On one GPU that is `micro_batch=8,
  grad_accum=8`, which is what the config sets.
- **If other users share the GPU.** Run `nvidia-smi` before each run and check the free
  memory. If it is too low for `micro_batch=8`, halve `micro_batch` and double
  `grad_accum` to keep the effective batch of 64. Do not stop or kill other
  users' jobs to make room. Wait, or use a smaller `micro_batch`.
- `finetune` returns a summary dict — **save it**. The snippet above writes it
  to `run_summary.json` in each output directory. It is the run fingerprint for
  the model card.

Then confirm it loads back:

```bash
.venv/bin/python -c "
import laya
a = laya.load('models/laya/en-v0.1.0')
print(a.predict('A video claims the PM resigned today.', {'misinfo': {'type':'noul','instructions':'Does this text make a factual claim that is false?'}}))
"
```

---

## 6. The part that is not written yet: the comparison

This is the deliverable, and it needs code that does not exist. Three pieces:

**6a. Score the Indian eval set with all three models.** `data/factchecks/`
holds 96 claims with `claim`, `claim_lang` and `label` (all `false`). Score
`claim` with Laya (English checkpoint for `en`, multilingual for `hi`; report
the base multilingual checkpoint on `hi` as well, see step 5), with the
`misinfo` roberta fine-tune, and with TF-IDF + logistic regression from
`modeling/eval/baselines.py`.

**6b. Compute lift, not bare recall.** A model that answers `false` to
everything scores **recall 1.0** on a positives-only set. So:

```
lift = recall on the 96 known-false claims / flag rate on the unlabelled corpus
```

Take the flag rate over `data/normalized/**` at the same threshold. Two
requirements for it:

- **Exclude the fact-check publishers from that corpus.** Their articles are the
  source of the 96 claims. A flag rate that includes them measures the model on
  its own answer key and is inflated.
- **Use titles for the primary flag rate.** The claims are short, so a
  title is the like-for-like input. Full text is a second value, reported next
  to it, not in place of it. 90% recall
at a 10% flag rate is a finding; 90% recall at an 85% flag rate is a model that
learned to say "false", and the recall figure alone cannot tell them apart.
Report both numbers side by side, always.

**6c. Report English and Hindi separately.** 83 of the 96 claims are English and
12 are Hindi, so the Hindi estimate rests on 12 items — wide intervals, and the
interval must be shown rather than the point estimate alone. `misinfo` and
TF-IDF cannot score the Hindi subset at all; that asymmetry *is* the result, and
"no score" is the honest entry in that cell, not a zero.

Reuse `modeling/eval/metrics.py` for the confidence intervals rather than
writing new ones, and `modeling/eval/report.py` for the artifact layout, so the
output lands next to the existing model cards in the shape the repo already uses.

---

## 7. Operating point and shipping

`configs/models.yaml` sets `decision_threshold: null` and
`precision_target: 0.85` deliberately. Same discipline as `bot` and `deepfake`:
a false "misinformation" costs more than a miss, so the threshold comes off the
PR curve at a precision target and is **re-derived for every checkpoint**. A
threshold copied from another run belongs to that run.

Weights never enter git — `/models/` is gitignored (anchored, see
`tests/test_repo_layout.py`). Push the checkpoint to a **private** HF repo and
set `HF_REPO` / `HF_TOKEN`; `modeling/registry.py` resolves
`MODELS_DIR → HF_REPO → GDRIVE_DIR`, first hit wins, and a miss writes honest
nulls rather than guessing.

---

## 8. If it should not ship

The bar the repo already set for `misinfo` applies here unchanged: *if the
transformer fails to clear TF-IDF it should not ship, because it costs orders of
magnitude more inference for no measured gain.*

Laya has one exemption from that bar and only one — **it scores Hindi, and the
other two cannot.** If Laya loses on English and wins on Hindi, say exactly
that; it is a more interesting finding than a model that wins everywhere, and it
is the one this corpus was rebuilt to make measurable.
