.PHONY: clean install test lint lint-fix test test-coverage test-total format format-fix show-resourses start-project

clean:
	rm -rf *.egg-info .pytest-cache/ .ruff-cache/ coverage.xml


install:
	uv sync


lint:
	uv run ruff check .


lint-fix:
	uv run ruff check --fix .


format:
	uv run ruff format --check .


format-fix:
	uv run ruff format .


test:
	uv run pytest


test-coverage:
	uv run pytest --cov=scripts --cov-report=xml:coverage.xml


test-total:
	uv run pytest --cov=scripts


show-resourses:
	uv run scalene --html --outfile profile.html main.py 


start-project:
	uv run main.py