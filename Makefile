.PHONY: install test format run-api clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Format code
format:
	black src/ && isort src/

# Run API server
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Clean temporary files
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

# Run all checks
check: test format