.PHONY: install lint format test clean compile typecheck

# Compile dependency lock files
compile:
	uv pip compile pyproject.toml --upgrade -o requirements.txt
	uv pip compile pyproject.toml --upgrade --extra dev -o requirements-dev.txt

# Install development dependencies from lockfile
install:
	uv pip install -r requirements-dev.txt

# Run linting checks
lint:
	ruff format . --check
	ruff check .
	python -m pyright

# Format code
format:
	ruff format .
	ruff check . --fix


# Run tests with coverage
test:
	pytest --cov=src tests/

build:
	uv build

publish:
	uv publish

