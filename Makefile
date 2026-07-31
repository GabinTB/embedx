# Local mirror of the CI gate (.github/workflows/ci.yml).

.PHONY: lint fmt test gate

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/embedx

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -m "not gpu"

gate: lint test
