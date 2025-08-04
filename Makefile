.PHONY: help install install-dev test test-unit test-integration lint format type-check clean setup-dev run-dev run-prod docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup-dev     Set up development environment"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean up temporary files"
	@echo "  run-dev       Run development server"
	@echo "  run-prod      Run production server"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

setup-dev:
	python scripts/setup_dev.py

# Testing
test:
	pytest -v --cov=stock_analysis_system --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

# Code quality
lint:
	flake8 stock_analysis_system tests
	black --check stock_analysis_system tests
	isort --check-only stock_analysis_system tests

format:
	black stock_analysis_system tests
	isort stock_analysis_system tests

type-check:
	mypy stock_analysis_system

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development server
run-dev:
	DB_HOST=localhost DB_PORT=5432 DB_NAME=stock_analysis DB_USER=postgres DB_PASSWORD=password uvicorn stock_analysis_system.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	DB_HOST=localhost DB_PORT=5432 DB_NAME=stock_analysis DB_USER=postgres DB_PASSWORD=password uvicorn stock_analysis_system.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Server management
start-server:
	python start_server.py

test-api:
	python test_api.py

# Docker services
docker-up:
	sudo docker-compose up -d postgres redis

docker-down:
	sudo docker-compose down

docker-status:
	sudo docker-compose ps

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(MESSAGE)"

# Docker
docker-build:
	docker build -t stock-analysis-system .

docker-run:
	docker run -p 8000:8000 stock-analysis-system

# Celery
celery-worker:
	celery -A stock_analysis_system.celery worker --loglevel=info

celery-beat:
	celery -A stock_analysis_system.celery beat --loglevel=info

celery-flower:
	celery -A stock_analysis_system.celery flower