PYTHON ?= python3

.PHONY: install test lint build clean

install:
	$(PYTHON) -m pip install -e '.[all,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
