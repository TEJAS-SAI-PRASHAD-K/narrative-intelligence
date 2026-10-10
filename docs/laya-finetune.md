# Fine-tuning Laya on a GPU box

What to run, in order, on a machine with a CUDA GPU. Written for a single A100;
the notes say where a smaller card changes something.

The deliverable is **not a Laya checkpoint**. It is the three-way comparison —
Laya vs the roberta-base fine-tune vs TF-IDF + logistic regression, on identical
splits through one eval harness — plus Laya's Hindi numbers, which neither of
the other two can produce. A model nobody compared to the baseline is not a
result.

---

## 0. Before you start: what is and isn't done

| | state |
|---|---|
| `modeling/text/laya_clf.py` | written, 24 tests, **never run against the real weights** |
| `configs/models.yaml → laya` | written: routing, question, train knobs, precision target |
| Training JSONL exporter | written (`LayaTrainingExporter`), tested |
| `data/benchmarks/` | **absent.** Nothing has been downloaded |
| The fine-tune | not run |
| The eval harness for the three-way comparison | **not written** — see step 6 |

Everything up to step 5 is runnable now. Step 6 is the part that still needs
code, and it is the part that produces the result.

---

## 1. Environment

```bash
python3 --version   # 3.10+ required by laya; 3.11-3.13 is the safe range
nvidia-smi
```

```bash
git clone https://github.com/TEJAS-SAI-PRASHAD-K/narrative-intelligence.git
cd narrative-intelligence
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[modeling,laya,dev]"
```

Do **not** reuse the laptop venv. That one is Python 3.14 with neither torch nor
transformers installed, and `.[laya]` would pull ~2 GB into it.

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
.venv/bin/python -m pip install -e ".[sources]"   # kaggle, needed by the fetcher
make benchmarks
```

Fetches LIAR, CoAID and FakeNewsNet into `data/benchmarks/` with checksums.

Two caveats carried over from the Phase 1 README, both still true:

- **FakeNewsNet ships ids and a crawler, not article text.** Budget time for the
  crawl, or train without it.
- **CoAID's tweet content cannot be recollected** in this project.

LIAR alone is enough to get a first number, and it is the one `misinfo` was
trained on — which is what makes the comparison like-for-like.

---

## 3. Why the training data is the benchmarks and not the Indian verdicts

`data/factchecks/` is **evaluation only**. Published fact-checks are
single-class — a measured 96 `false` and 0 `true` — because debunking is news
and confirming is not. Nothing can be trained on one class.

So: train on LIAR (US political statements), evaluate on Indian
recontextualised media. The domain gap is not a flaw in this design, it is the
quantity being measured. See the README section "Fact-check supervision".

---

## 4. Export the training JSONL

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from modeling.datasets.liar import Liar
from modeling.datasets.splits import group_train_val_test
from modeling.text.laya_clf import LayaTrainingExporter

loaded = Liar().load()            # LoadedDataset: .frame, .group_col, .label_col
print(loaded.describe())
print("refused rows by reason:", loaded.dropped)

# Group-aware split, NOT a random one. LIAR repeats speakers, so a random
# split leaks a speaker's style across the boundary and inflates every metric.
# This is the same splitter misinfo uses, which is what makes the comparison
# like-for-like rather than merely similar.
train, val, test = group_train_val_test(
    loaded.frame, group_col=loaded.group_col, label_col=loaded.label_col
)

exporter = LayaTrainingExporter()
for name, frame in (("train", train), ("val", val)):
    n = exporter.export(frame, Path(f"data/laya/{name}.jsonl"), label_column=loaded.label_col)
    print(name, n)
# `test` is NOT exported. It is held for step 6 and must never reach finetune.
PY
```

> Check `group_train_val_test`'s real signature before running — it is in
> `modeling/datasets/splits.py` and may take its arguments positionally. The
> exporter itself only needs a DataFrame with a text column and a label column
> where 1 means misinformation.
>
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

## 5. Fetch the base checkpoint, then fine-tune

`finetune` takes a **local checkpoint directory**, not a Hub id:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("convaiinnovations/laya", local_dir="models/laya/base"))
PY
```

For the multilingual checkpoint add `allow_patterns=["multilingual/*"]` or pass
`subfolder="multilingual"` at load time; `configs/models.yaml` already records
which language routes to which.

```bash
.venv/bin/python - <<'PY'
from laya.train import TrainConfig, finetune
from modeling.config import module_config

cfg = module_config("laya")["train"]
summary = finetune(
    "data/laya/train.jsonl",
    "models/laya/base",
    "models/laya/v0.1.0",
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
    ),
)
print(summary)
PY
```

Notes that matter:

- **Fine-tuning is mandatory.** Vendor-reported, the base checkpoint scores
  0.362 on typed-decisions against a 0.461 majority-class baseline — *worse than
  guessing*. Run the zero-shot number too, but report it as a baseline.
- **Calibration happens inside `finetune`.** It holds out `calib_frac` (0.1) to
  fit one temperature per (question type, option count). Vendor-reported mean
  ECE 0.466 → 0.081. This is why the module has no separate `Calibrator`, unlike
  `misinfo`.
- The defaults assume the published notebook's effective batch of 64
  (8 × 2 GPUs × 4 accumulation). On one A100 that is `micro_batch=8,
  grad_accum=8`, which is what the config sets. On a smaller card, halve
  `micro_batch` and double `grad_accum` to keep the effective batch.
- `finetune` returns a summary dict — **save it**. It is the run fingerprint for
  the model card.

Then confirm it loads back:

```bash
.venv/bin/python -c "
import laya
a = laya.load('models/laya/v0.1.0')
print(a.predict('A video claims the PM resigned today.', {'misinfo': {'type':'noul','instructions':'Does this text make a factual claim that is false?'}}))
"
```

---

## 6. The part that is not written yet: the comparison

This is the deliverable, and it needs code that does not exist. Three pieces:

**6a. Score the Indian eval set with all three models.** `data/factchecks/`
holds 96 claims with `claim`, `claim_lang` and `label` (all `false`). Score
`claim` with Laya (English checkpoint for `en`, multilingual for `hi`), with the
`misinfo` roberta fine-tune, and with TF-IDF + logistic regression from
`modeling/eval/baselines.py`.

**6b. Compute lift, not bare recall.** A model that answers `false` to
everything scores **recall 1.0** on a positives-only set. So:

```
lift = recall on the 96 known-false claims / flag rate on the unlabelled corpus
```

Take the flag rate over `data/normalized/**` at the same threshold. 90% recall
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
