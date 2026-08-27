.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup data fetch normalize validate stats test lint fmt benchmarks clean clean-data \
	up down logs ps migrate revision downgrade load reload verify seed openapi api-shell db-shell

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

test: ## Run the test suite (no live network calls, no GPU)
	$(BIN)/pytest

test-api: ## Run the suite including the Postgres-backed API tests
	@docker inspect ni-test-pg >/dev/null 2>&1 || \
		docker run -d --rm --name ni-test-pg -e POSTGRES_USER=narrative \
			-e POSTGRES_PASSWORD=test -e POSTGRES_DB=narrative_test -e TZ=UTC \
			-p 127.0.0.1:55433:5432 pgvector/pgvector:pg16 >/dev/null
	@until docker exec ni-test-pg pg_isready -U narrative >/dev/null 2>&1; do sleep 1; done
	@docker exec ni-test-pg psql -qU narrative -d narrative_test \
		-c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null
	TEST_DATABASE_URL=postgresql+psycopg://narrative:test@127.0.0.1:55433/narrative_test \
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

# --- Phase 4: backend, persistence & API ------------------------------------
# Compose v2 ships either as a `docker compose` subcommand or as a standalone
# `docker-compose` binary depending on how Docker was installed. Detect rather
# than assume: `make up` failing with "unknown command" on a machine that has a
# perfectly good Compose is a bad first experience.
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# On macOS, torch and xgboost each ship their own OpenMP runtime and the two
# fight: the process dies mid-inference with no traceback and no exit message.
# Phase 2 documents the import-order half of the fix (xgboost before torch, in
# modeling/__init__.py); this is the other half. It serialises OpenMP, which
# costs throughput on a CPU inference worker and buys not segfaulting.
# Harmless on Linux, where there is only one runtime.
export OMP_NUM_THREADS ?= 1

setup-api: ## Install the Phase 4 backend dependencies into the venv
	$(BIN)/pip install -e ".[api,dev]"

up: ## Bring the whole stack up (db, redis, api, worker, beat)
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - fill in POSTGRES_PASSWORD and API_KEY_PEPPER, then rerun." && exit 1)
	$(COMPOSE) up --build -d
	@echo "API on http://localhost:$${API_HOST_PORT:-8000}/docs"

up-deps: ## Just Postgres and Redis, for running uvicorn on the host
	$(COMPOSE) up -d db redis

down: ## Stop the stack, keep the volumes
	$(COMPOSE) down

nuke: ## DESTRUCTIVE: stop the stack and delete the database volume
	@echo "This deletes the Postgres volume. The Parquet corpus under data/ survives."
	@read -p "Type 'yes' to confirm: " ok && [ "$$ok" = "yes" ] && $(COMPOSE) down -v || echo "aborted"

logs: ## Tail every service
	$(COMPOSE) logs -f --tail=100

ps: ## Service status
	$(COMPOSE) ps

api-shell: ## Shell inside the api container
	$(COMPOSE) run --rm --entrypoint bash api

db-shell: ## psql inside the db container
	$(COMPOSE) exec db psql -U $${POSTGRES_USER:-narrative} -d $${POSTGRES_DB:-narrative}

# --- schema ----------------------------------------------------------------
migrate: ## Apply every migration
	$(BIN)/alembic upgrade head

downgrade: ## Roll back one migration
	$(BIN)/alembic downgrade -1

revision: ## Autogenerate a migration: make revision M="add foo"
	$(BIN)/alembic revision --autogenerate -m "$(M)"

migrate-check: ## Prove the migration chain round-trips (downgrade base, upgrade head)
	$(BIN)/alembic downgrade base
	$(BIN)/alembic upgrade head

# --- data ------------------------------------------------------------------
load: ## Load the Phase 1 Parquet corpus into Postgres
	$(BIN)/python -m app.etl.cli load --project $(P)

reload: ## Force-reload a project's corpus
	$(BIN)/python -m app.etl.cli reload --project $(P) --force

verify: ## Reconcile manifest <-> Parquet <-> Postgres row counts
	$(BIN)/python -m app.etl.cli verify --project $(P)

seed: ## Build the curated demo project (no network, no model warm-up)
	$(BIN)/python -m app.etl.cli seed --slug $(or $(P),demo)

# --- contract --------------------------------------------------------------
openapi: ## Regenerate openapi.json from the code
	$(BIN)/python -m app.openapi_export openapi.json

openapi-check: ## Fail if openapi.json has drifted from the code
	$(BIN)/python -m app.openapi_export --check openapi.json

serve: ## Run the API on the host against `make up-deps`
	$(BIN)/uvicorn app.main:app --reload --port $${API_HOST_PORT:-8000}
