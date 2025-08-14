.PHONY: install train api format lint clean

# Project setup
install:
	poetry install

# Generate dataset for CRNN OCR model
synthesize:
	poetry run python scripts/synthesize_dataset.py

# Train CRNN OCR model
train:
	poetry run python scripts/train_crnn.py

# Run FastAPI server
api:
	poetry run uvicorn app.main:app --reload

# Format code using Black and isort
format:
	poetry run black .
	poetry run isort .

# Lint (optional: expand with flake8/mypy if needed)
lint:
	poetry run black --check .
	poetry run isort --check-only .

# Clean up model artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache *.png *.h5 crnn_model_inference
