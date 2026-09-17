.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size 2000

debug:
	uv run python -m pdb -m src index --max_chunk_size 2000

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache data/processed/*

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src
