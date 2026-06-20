# psobb-studio — developer tasks. Requires Python 3.11 (and `make`).
# `make help` lists targets. Override PORT/DATA on the command line, e.g.
#   make run PORT=9000 DATA=/games/PSOBB/data

PORT ?= 8765
DATA  ?= $(HOME)/PSOBB.IO/data
PY    ?= python

.DEFAULT_GOAL := help
.PHONY: help install dev run smoke test lint clean

help: ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## install the app + runtime deps (editable)
	$(PY) -m pip install -e .

dev: ## install app + developer tooling (pytest, ruff)
	$(PY) -m pip install -e ".[dev]"

run: ## launch the server (override PORT= / DATA=)
	PSO_DATA_DIR="$(DATA)" $(PY) -m uvicorn server:app --host 127.0.0.1 --port $(PORT) --reload

smoke: ## fresh import + boot smoke test (the same check CI runs)
	$(PY) scripts/smoke_test.py

test: ## run the unit/integration test suite
	$(PY) -m pytest

lint: ## static checks (ruff)
	$(PY) -m ruff check .

clean: ## remove caches
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
