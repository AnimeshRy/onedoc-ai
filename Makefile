help:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

activate: ## Reminder to activate the virtual environment
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

install: ## Install requirements from pip
	uv run

dev-install: ## Install requirements from pip in dev environment
	uv run --dev

run: ## Run the service
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
