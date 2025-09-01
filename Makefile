# Makefile for TTV Pipeline API Docker deployment
# Provides convenient shortcuts for common Docker operations

.PHONY: help build up down restart logs status clean health test dev prod gpu debug setup setup-monitoring docs

# Default environment
ENV ?= production

# Docker Compose files
COMPOSE_DEV = -f docker-compose.yml -f docker-compose.dev.yml
COMPOSE_PROD = -f docker-compose.yml -f docker-compose.prod.yml

# Environment-specific compose files
ifeq ($(ENV),development)
	COMPOSE_FILES = $(COMPOSE_DEV)
	ENV_FILE = .env.dev
else ifeq ($(ENV),dev)
	COMPOSE_FILES = $(COMPOSE_DEV)
	ENV_FILE = .env.dev
else ifeq ($(ENV),production)
	COMPOSE_FILES = $(COMPOSE_PROD)
	ENV_FILE = .env.prod
else ifeq ($(ENV),prod)
	COMPOSE_FILES = $(COMPOSE_PROD)
	ENV_FILE = .env.prod
else
	COMPOSE_FILES = -f docker-compose.yml
	ENV_FILE = .env
endif

# Default target
help: ## Show this help message
	@echo "TTV Pipeline API Docker Management"
	@echo "=================================="
	@echo ""
	@echo "Usage: make [target] [ENV=environment]"
	@echo ""
	@echo "Environments:"
	@echo "  development, dev  - Development environment with hot reload"
	@echo "  production, prod  - Production environment with optimizations"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build Docker images
	@echo "Building images for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) build

up: ## Start services
	@echo "Starting services for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) up -d

down: ## Stop services
	@echo "Stopping services for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) down

restart: ## Restart services
	@echo "Restarting services for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) restart

logs: ## Show service logs
	@echo "Showing logs for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) logs -f

status: ## Show service status
	@echo "Service status for $(ENV) environment:"
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) ps

clean: ## Clean up containers, networks, and volumes
	@echo "Cleaning up $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) down -v --remove-orphans
	docker system prune -f

health: ## Run health checks
	@echo "Running health checks for $(ENV) environment..."
	./scripts/health-check.sh $(ENV)

test: ## Run tests in containers
	@echo "Running tests for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) exec api pytest tests/

# Environment-specific shortcuts
dev: ## Start development environment
	@$(MAKE) up ENV=development

prod: ## Start production environment
	@$(MAKE) up ENV=production

gpu: ## Start with GPU support
	@echo "Starting with GPU support for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile gpu up -d

debug: ## Start with debug services
	@echo "Starting with debug services for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile debug up -d

# Development helpers
dev-build: ## Build development images
	@$(MAKE) build ENV=development

dev-logs: ## Show development logs
	@$(MAKE) logs ENV=development

dev-shell: ## Access development API container shell
	docker compose $(COMPOSE_DEV) --env-file .env.dev exec api bash

dev-test: ## Run tests in development environment
	@$(MAKE) test ENV=development

# Production helpers
prod-build: ## Build production images
	@$(MAKE) build ENV=production

prod-logs: ## Show production logs
	@$(MAKE) logs ENV=production

prod-deploy: ## Deploy to production (build + up)
	@$(MAKE) build ENV=production
	@$(MAKE) up ENV=production

# Maintenance
update: ## Update images and restart services
	@echo "Updating images for $(ENV) environment..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) pull
	@$(MAKE) build ENV=$(ENV)
	@$(MAKE) restart ENV=$(ENV)

backup: ## Backup Redis data and configuration
	@echo "Creating backup for $(ENV) environment..."
	mkdir -p backup
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) exec redis redis-cli BGSAVE
	docker cp $$(docker compose $(COMPOSE_FILES) ps -q redis):/data/dump.rdb ./backup/redis-$(ENV)-$$(date +%Y%m%d-%H%M%S).rdb
	tar -czf backup/config-$(ENV)-$$(date +%Y%m%d-%H%M%S).tar.gz .env* docker-compose*.yml config/ credentials/ || true

# Scaling
scale-workers: ## Scale workers (usage: make scale-workers REPLICAS=4)
	@echo "Scaling workers to $(REPLICAS) replicas..."
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) up -d --scale worker=$(REPLICAS)

# Monitoring
metrics: ## Show API metrics
	@echo "Fetching metrics from API..."
	curl -s http://localhost:8000/metrics || echo "Metrics endpoint not available"

api-health: ## Check API health endpoint
	@echo "Checking API health..."
	curl -f http://localhost:8000/healthz && echo " ✓ API is healthy" || echo " ✗ API health check failed"

redis-info: ## Show Redis information
	@echo "Redis information:"
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) exec redis redis-cli info server

# Setup helpers
setup-dev: ## Set up development environment
	@echo "Setting up development environment..."
	cp .env.dev .env || true
	mkdir -p credentials certs backup
	@echo "Development environment ready. Edit .env with your configuration."

setup-prod: ## Set up production environment
	@echo "Setting up production environment..."
	cp .env.prod .env || true
	mkdir -p credentials certs backup
	@echo "Production environment ready. Edit .env with your configuration and add certificates."

# Quick commands
quick-dev: setup-dev dev-build dev ## Quick development setup and start
quick-prod: setup-prod prod-build prod ## Quick production setup and start

# Environment Setup
setup: ## Setup environment (usage: make setup ENV=dev)
	@echo "Setting up $(ENV) environment..."
	./scripts/setup-environment.sh $(ENV)

setup-monitoring: ## Setup monitoring stack (usage: make setup-monitoring STACK=prometheus)
	@echo "Setting up monitoring stack: $(STACK)"
	./scripts/setup-monitoring.sh $(STACK)

# Documentation
docs: ## Open documentation
	@echo "Opening documentation..."
	@if command -v open >/dev/null 2>&1; then \
		open docs/README.md; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/README.md; \
	else \
		echo "Documentation available at: docs/README.md"; \
	fi