# FinSight Financial Sentiment Analysis
# Makefile for development, testing, and deployment

.PHONY: help install install-dev test lint format check-all
.PHONY: run-local run-docker build-docker
.PHONY: data-collect train-model evaluate-model clean setup-env
.PHONY: docker-up docker-down docker-logs monitor

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := finsight
VERSION := 1.0.0
ENV_FILE := .env

# Colors for output
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)FinSight Financial Sentiment Analysis$(RESET)"
	@echo "$(BLUE)==================================$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Quick Start:$(RESET)"
	@echo "  1. make setup-env          # Set up environment"
	@echo "  2. make install-dev         # Install dependencies"
	@echo "  3. make docker-up           # Start development stack"
	@echo "  4. make train-model         # Train the ML model"
	@echo "  5. make run-local           # Start the API server"

## Environment Setup
setup-env: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	@if [ ! -f $(ENV_FILE) ]; then \
		cp env.example $(ENV_FILE); \
		echo "$(YELLOW)Created .env file from template. Please edit with your API keys.$(RESET)"; \
	else \
		echo "$(YELLOW).env file already exists.$(RESET)"; \
	fi
	@$(PYTHON) -m venv venv || echo "$(RED)Failed to create virtual environment$(RESET)"
	@echo "$(GREEN)Virtual environment created. Activate with: source venv/bin/activate$(RESET)"

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(RESET)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Production dependencies installed successfully!$(RESET)"

install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	@$(PIP) install -r requirements-dev.txt
	@pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

## Code Quality
lint: ## Run code linting
	@echo "$(GREEN)Running linters...$(RESET)"
	@echo "$(YELLOW)Running flake8...$(RESET)"
	@flake8 src/ tests/ scripts/ || echo "$(RED)Flake8 found issues$(RESET)"
	@echo "$(YELLOW)Running mypy...$(RESET)"
	@mypy src/insight/ || echo "$(RED)MyPy found issues$(RESET)"
	@echo "$(YELLOW)Running bandit security scan...$(RESET)"
	@bandit -r src/ || echo "$(RED)Bandit found security issues$(RESET)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	@black src/ tests/ scripts/
	@isort src/ tests/ scripts/
	@echo "$(GREEN)Code formatting completed!$(RESET)"

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(RESET)"
	@bandit -r src/ || echo "$(RED)Bandit found security issues$(RESET)"
	@safety check || echo "$(RED)Safety check found vulnerabilities$(RESET)"

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(RESET)"
	@pytest tests/ -v --cov=src/insight --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Tests completed! Coverage report: htmlcov/index.html$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	@pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	@pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running end-to-end tests...$(RESET)"
	@pytest tests/e2e/ -v

check-all: lint test ## Run all quality checks

## Data and Model Operations
data-collect: ## Collect training data from all sources
	@echo "$(GREEN)Collecting training data...$(RESET)"
	@$(PYTHON) scripts/collect_data.py --limit 10000 --output data/raw/
	@echo "$(GREEN)Data collection completed!$(RESET)"

preprocess-data: ## Preprocess collected data
	@echo "$(GREEN)Preprocessing data...$(RESET)"
	@$(PYTHON) scripts/preprocess_data.py --input data/raw/ --output data/processed/
	@echo "$(GREEN)Data preprocessing completed!$(RESET)"

train-model: ## Train the sentiment analysis model
	@echo "$(GREEN)Training sentiment analysis model...$(RESET)"
	@$(PYTHON) scripts/train_model.py \
		--epochs 5 \
		--batch-size 16 \
		--learning-rate 2e-5 \
		--save-path data/models/sentiment_model
	@echo "$(GREEN)Model training completed!$(RESET)"

evaluate-model: ## Evaluate the trained model
	@echo "$(GREEN)Evaluating model performance...$(RESET)"
	@$(PYTHON) scripts/evaluate_model.py \
		--model-path data/models/sentiment_model \
		--test-data data/processed/test.csv \
		--output-path data/models/evaluation/
	@echo "$(GREEN)Model evaluation completed! Check data/models/evaluation/$(RESET)"

## Local Development
run-local: ## Run the API server locally
	@echo "$(GREEN)Starting local API server...$(RESET)"
	@uvicorn src.insight.api.app:create_app --host 0.0.0.0 --port 8000 --reload

run-prod: ## Run the API server in production mode
	@echo "$(GREEN)Starting production API server...$(RESET)"
	@uvicorn src.insight.api.app:create_app --host 0.0.0.0 --port 8000 --workers 4

jupyter: ## Start Jupyter notebook server
	@echo "$(GREEN)Starting Jupyter notebook server...$(RESET)"
	@jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser notebooks/

tensorboard: ## Start TensorBoard for model monitoring
	@echo "$(GREEN)Starting TensorBoard...$(RESET)"
	@tensorboard --logdir=data/models/logs --port=6006

mlflow-ui: ## Start MLflow UI
	@echo "$(GREEN)Starting MLflow UI...$(RESET)"
	@mlflow ui --host 0.0.0.0 --port 5000

## Docker Operations
build-docker: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(RESET)"
	@$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	@$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker image built successfully!$(RESET)"

run-docker: docker-up ## Start development stack with Docker Compose

docker-up: ## Start all services with Docker Compose
	@echo "$(GREEN)Starting Docker Compose stack...$(RESET)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started! API: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin)$(RESET)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(RESET)"

docker-down: ## Stop all Docker services
	@echo "$(GREEN)Stopping Docker Compose stack...$(RESET)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped!$(RESET)"

docker-logs: ## Show Docker Compose logs
	@$(DOCKER_COMPOSE) logs -f

docker-build: ## Build services with Docker Compose
	@echo "$(GREEN)Building services with Docker Compose...$(RESET)"
	@$(DOCKER_COMPOSE) build

docker-clean: ## Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(RESET)"
	@$(DOCKER) system prune -f
	@$(DOCKER) volume prune -f
	@echo "$(GREEN)Docker cleanup completed!$(RESET)"

## Monitoring and Observability
setup-monitoring: ## Set up monitoring infrastructure
	@echo "$(GREEN)Setting up monitoring infrastructure...$(RESET)"
	@mkdir -p infrastructure/monitoring
	@echo "# Prometheus configuration" > infrastructure/monitoring/prometheus.yml
	@echo "global:" >> infrastructure/monitoring/prometheus.yml
	@echo "  scrape_interval: 15s" >> infrastructure/monitoring/prometheus.yml
	@echo "scrape_configs:" >> infrastructure/monitoring/prometheus.yml
	@echo "  - job_name: 'insight-api'" >> infrastructure/monitoring/prometheus.yml
	@echo "    static_configs:" >> infrastructure/monitoring/prometheus.yml
	@echo "      - targets: ['insight-api:8000']" >> infrastructure/monitoring/prometheus.yml
	@echo "$(GREEN)Monitoring configuration created!$(RESET)"

monitor: ## Open monitoring dashboards
	@echo "$(GREEN)Opening monitoring dashboards...$(RESET)"
	@open http://localhost:3000 || echo "Grafana: http://localhost:3000"
	@open http://localhost:9090 || echo "Prometheus: http://localhost:9090"

## Docker Deployment
deploy: build-docker ## Build and tag Docker image for deployment
	@echo "$(GREEN)Docker image ready for deployment$(RESET)"
	@$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Tagged $(PROJECT_NAME):latest$(RESET)"

## Utilities
clean: ## Clean up generated files and caches
	@echo "$(GREEN)Cleaning up...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ htmlcov/ .coverage security-report.json
	@echo "$(GREEN)Cleanup completed!$(RESET)"

backup-data: ## Backup important data
	@echo "$(GREEN)Backing up data...$(RESET)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r data/models/ backups/$(shell date +%Y%m%d_%H%M%S)/models/ || true
	@cp -r data/processed/ backups/$(shell date +%Y%m%d_%H%M%S)/processed/ || true
	@echo "$(GREEN)Data backup completed!$(RESET)"

status: ## Show project status
	@echo "$(BLUE)Project Status$(RESET)"
	@echo "$(BLUE)==============$(RESET)"
	@echo "Python version: $(shell $(PYTHON) --version 2>&1)"
	@echo "Virtual environment: $(shell echo $$VIRTUAL_ENV)"
	@echo "Docker status: $(shell $(DOCKER) --version 2>&1)"
	@echo "Docker Compose status: $(shell $(DOCKER_COMPOSE) --version 2>&1)"
	@echo ""
	@echo "$(GREEN)Services Status:$(RESET)"
	@$(DOCKER_COMPOSE) ps || echo "Docker Compose not running"

## Development Workflow
dev-setup: setup-env install-dev ## Complete development setup
	@echo "$(GREEN)Development environment setup completed!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Edit .env file with your API keys"
	@echo "  2. Run 'make docker-up' to start services"
	@echo "  3. Run 'make train-model' to train the model"
	@echo "  4. Run 'make run-local' to start the API"

full-pipeline: data-collect train-model evaluate-model ## Run complete ML pipeline
	@echo "$(GREEN)Full ML pipeline completed!$(RESET)"

## Development Tools
notebook: ## Start Jupyter notebook
	@echo "$(GREEN)Starting Jupyter notebook...$(RESET)"
	@jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser notebooks/

docs: ## Open API documentation
	@echo "$(GREEN)API documentation available at:$(RESET)"
	@echo "http://localhost:8000/docs"
