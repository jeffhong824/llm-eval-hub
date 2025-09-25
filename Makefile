.PHONY: help install run test lint format clean build up down logs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

run: ## Run the application locally
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	ruff check .
	black --check .
	mypy .

format: ## Format code
	black .
	ruff check --fix .

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

build: ## Build Docker image
	docker build -t llm-eval-hub .

up: ## Start services with docker-compose
	docker compose up -d

down: ## Stop services
	docker compose down

logs: ## Show logs
	docker compose logs -f

setup: ## Initial setup
	mkdir -p data/uploads results artifacts
	cp env.example .env
	@echo "Please edit .env file with your API keys"

install-deps: ## Install Python dependencies
	pip install -r requirements.txt

start-local: ## Start application locally (no Docker)
	python start-local.py

start-docker: ## Start with Docker Compose
	docker compose up -d

check-docker: ## Check Docker status
	docker --version
	docker compose version
	docker info

help-docker: ## Show Docker help
	@echo "Docker Setup Help:"
	@echo "1. Start Docker Desktop"
	@echo "2. Wait for it to fully load"
	@echo "3. Run: make start-docker"
	@echo "4. Access: http://localhost:8000"
