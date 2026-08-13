.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup data fetch normalize validate stats test lint fmt benchmarks clean clean-data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip setuptools wheel

setup: $(BIN)/python ## Create venv and install everything (core + sources + dev)
	$(BIN)/pip install -e ".[dev,notebook]"
	@echo ""
	@echo "Core install complete. Installing source adapters (heavy, may take a while)..."
	@$(BIN)/pip install -e ".[sources]" || \
		echo "WARNING: one or more optional source deps failed to install. Core pipeline still works; affected adapters will skip with a clear message. See README > Troubleshooting."
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - fill it in.")

data: ## Full rebuild: fetch every available source -> normalized parquet + manifest
	$(BIN)/python -m ingest.cli fetch-all

fetch: ## Fetch a single source: make fetch SOURCE=mastodon
	$(BIN)/python -m ingest.cli fetch $(SOURCE)

validate: ## Re-validate the whole normalized corpus against the schema
	$(BIN)/python -m ingest.cli validate

stats: ## Per-source summary table of the corpus on disk
	$(BIN)/python -m ingest.cli stats

benchmarks: ## Download LIAR / FakeNewsNet / CoAID for Phase 2 (not used in Phase 1)
	$(BIN)/python scripts/download_benchmarks.py

# --- Phase 2: modeling & scoring -------------------------------------------
setup-modeling: ## Install the Phase 2 dependencies
	$(BIN)/pip install -e ".[modeling]"

warm-cache: ## Pre-download the auxiliary models so scoring runs work offline
	$(BIN)/python -m modeling.cli warm-cache

score: ## Score the corpus into data/scored/ (resumable, idempotent)
	$(BIN)/python -m modeling.cli score --all

score-demo: ## Score the committed fixtures: no network, no benchmarks, ~12s
	$(BIN)/python -m modeling.cli score --all --demo

train-misinfo: ## Train the misinformation classifier (needs a benchmark on disk)
	$(BIN)/python -m modeling.cli train misinfo

eval-report: ## Regenerate artifacts/eval/** from saved predictions, no retraining
	$(BIN)/python -m modeling.cli report

ablate: ## The module-ablation table
	$(BIN)/python -m modeling.cli ablate

notebooks: ## Regenerate the notebook skeletons from their build scripts
	$(BIN)/python notebooks/build_eda_notebook.py
	$(BIN)/python notebooks/build_phase2_notebooks.py

fixtures: ## Regenerate the committed test fixtures
	$(BIN)/python scripts/make_fixtures.py

test: ## Run the test suite (no live network calls)
	$(BIN)/pytest

lint: ## Lint
	$(BIN)/ruff check ingest modeling tests scripts

fmt: ## Auto-fix lint + format
	$(BIN)/ruff check --fix ingest modeling tests scripts
	$(BIN)/ruff format ingest modeling tests scripts

clean: ## Remove caches and build junk (keeps data/)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist

clean-data: ## DESTRUCTIVE: delete the entire local corpus
	@echo "This deletes data/ (raw + normalized + checkpoints + manifest)."
	@read -p "Type 'yes' to confirm: " ok && [ "$$ok" = "yes" ] && rm -rf data || echo "aborted"
