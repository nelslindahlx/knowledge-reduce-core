.PHONY: help install canary skill lint golden smoke-pr test ci local-ci audit-prs clean

ifneq ($(wildcard .venv/Scripts/python.exe),)
    DEFAULT_PYTHON := .venv/Scripts/python.exe
else ifneq ($(wildcard .venv/bin/python),)
    DEFAULT_PYTHON := .venv/bin/python
else
    DEFAULT_PYTHON := uv run --no-sync python
endif
PYTHON ?= $(DEFAULT_PYTHON)

help:
	@echo "Available commands (PYTHON=$(PYTHON)):"
	@echo "  make install    - Install editable package with dev and hermes extras"
	@echo "  make canary     - Run <10s import warning gate and tests/meta/ invariants"
	@echo "  make skill      - Run Hermes skill smoke check and manifest tests"
	@echo "  make lint       - Run ruff check and ruff format --check on governance/meta modules"
	@echo "  make golden     - Run Civic Honors book and multi-source factory golden benchmarks"
	@echo "  make smoke-pr   - Run canary + skill + lint + Stage 1 & Stage 2 test suites"
	@echo "  make test       - Run full pytest suite with 75% coverage gate"
	@echo "  make local-ci   - Run complete stratified local CI (zero GitHub Actions minutes)"
	@echo "  make audit-prs  - Verify zero open PRs and zero lingering remote feature branches"
	@echo "  make clean      - Remove build, cache, coverage, and stray test artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev,hermes]"

canary:
	$(PYTHON) -W error::Warning -c "import knowledge_graph_pkg"
	$(PYTHON) -m pytest tests/meta/ -q

skill:
	$(PYTHON) scripts/smoke_skill.py
	$(PYTHON) -m pytest tests/test_skill.py -q

lint:
	$(PYTHON) -m ruff check knowledge_graph_pkg tests scripts benchmarks
	$(PYTHON) -m ruff check --select E,F,W,I,B,UP,S,SIM,RUF --ignore E501,S101 scripts/github_pr_helper.py scripts/hooks tests/meta benchmarks tests/golden
	$(PYTHON) -m ruff format --check scripts/github_pr_helper.py scripts/hooks tests/meta benchmarks tests/golden

golden:
	$(PYTHON) benchmarks/civichonors_book_benchmark.py --strict-regression
	$(PYTHON) benchmarks/multisource_factory_benchmark.py --strict-regression

smoke-pr: canary skill lint
	$(PYTHON) scripts/run_suite.py --stage 1
	$(PYTHON) scripts/run_suite.py --stage 2

test:
	$(PYTHON) -m pytest --cov=knowledge_graph_pkg --cov-report=term-missing --cov-fail-under=75

ci: local-ci

local-ci: canary skill lint golden test
	@echo "Local CI run complete. Zero GitHub Action minutes consumed."

audit-prs:
	$(PYTHON) scripts/github_pr_helper.py --audit

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov/ .coverage coverage.xml
	rm -rf test_store_* test_watch_* test_watcher_*.db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
