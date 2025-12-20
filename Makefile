.PHONY: all build build-embedded test clean deps model lint run-test setup-python help

# Go parameters
GOCMD = go
GOBUILD = $(GOCMD) build
GOTEST = $(GOCMD) test
GOCLEAN = $(GOCMD) clean
GOMOD = $(GOCMD) mod
BINARY_NAME = qrdetect

# Python for model export
PYTHON = python3
VENV = .venv
PYTHON_VENV = $(VENV)/bin/python

# Model paths
# PyTorch model is cached in root models/ directory
MODEL_CACHE_DIR = models
MODEL_PT = $(MODEL_CACHE_DIR)/qrdet-s.pt
# ONNX model goes in pkg/qrdet/models/ for go:embed
MODEL_EMBED_DIR = pkg/qrdet/models
MODEL_ONNX = $(MODEL_EMBED_DIR)/qrdet-s.onnx
MODEL_URL = https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-s.pt

# Default target
all: deps model build test

# ------------------------------------------------------------------------------
# Model Management
# ------------------------------------------------------------------------------

# Download and export ONNX model
model: $(MODEL_ONNX)

$(MODEL_ONNX): $(MODEL_PT) scripts/export_model.py
	@echo "==> Exporting model to ONNX format..."
	@mkdir -p $(MODEL_EMBED_DIR)
	$(PYTHON_VENV) scripts/export_model.py --output $(MODEL_ONNX)
	@echo "==> Model exported to $(MODEL_ONNX)"

$(MODEL_PT):
	@echo "==> Downloading QRdet model..."
	@mkdir -p $(MODEL_CACHE_DIR)
	curl -L $(MODEL_URL) -o $(MODEL_PT)
	@echo "==> Model downloaded to $(MODEL_PT)"

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

deps:
	@echo "==> Downloading Go dependencies..."
	$(GOMOD) download
	$(GOMOD) tidy
	@echo "==> Dependencies installed"

# Setup Python virtual environment for model export
setup-python:
	@echo "==> Setting up Python virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install ultralytics onnx requests tqdm
	@echo "==> Python environment ready. Activate with: source $(VENV)/bin/activate"
	@echo "==> Note: onnxsim is optional (requires cmake). Model export works without it."

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

# Build the CLI (development mode, loads model from file)
build:
	@echo "==> Building $(BINARY_NAME) (dev mode)..."
	$(GOBUILD) -o $(BINARY_NAME) ./cmd/qrdetect
	@echo "==> Built $(BINARY_NAME)"

# Build with embedded model (requires model to exist)
build-embedded: $(MODEL_ONNX)
	@echo "==> Building $(BINARY_NAME) with embedded model..."
	$(GOBUILD) -tags embed_model -o $(BINARY_NAME) ./cmd/qrdetect
	@echo "==> Built $(BINARY_NAME) with embedded model"

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

test:
	@echo "==> Running tests..."
	$(GOTEST) -v ./...

test-coverage:
	@echo "==> Running tests with coverage..."
	$(GOTEST) -v -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "==> Coverage report: coverage.html"

bench:
	@echo "==> Running benchmarks..."
	$(GOTEST) -bench=. -benchmem ./pkg/qrdet/...

# ------------------------------------------------------------------------------
# Development
# ------------------------------------------------------------------------------

# Run detection on a test image
run-test: build
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make run-test INPUT=path/to/image.png [OUTPUT=output.png]"; \
		exit 1; \
	fi
	./$(BINARY_NAME) -input $(INPUT) -output $(or $(OUTPUT),output.png)

# Run detection with model path (for development)
run-dev: build
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make run-dev INPUT=path/to/image.png [OUTPUT=output.png]"; \
		exit 1; \
	fi
	./$(BINARY_NAME) -model $(MODEL_ONNX) -input $(INPUT) -output $(or $(OUTPUT),output.png)

# Lint the code
lint:
	@echo "==> Running linter..."
	golangci-lint run ./...

# Format the code
fmt:
	@echo "==> Formatting code..."
	$(GOCMD) fmt ./...

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------

clean:
	@echo "==> Cleaning build artifacts..."
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f coverage.out coverage.html
	@echo "==> Clean complete"

clean-all: clean
	@echo "==> Cleaning all artifacts including models..."
	rm -rf $(MODEL_CACHE_DIR)/*.onnx $(MODEL_CACHE_DIR)/*.pt
	rm -rf $(MODEL_EMBED_DIR)/*.onnx
	rm -rf $(VENV)
	@echo "==> Full clean complete"

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

help:
	@echo "QReader Go Port - Makefile targets"
	@echo ""
	@echo "Model Management:"
	@echo "  model           Download QRdet weights and export to ONNX"
	@echo "  setup-python    Setup Python venv for model export"
	@echo ""
	@echo "Build:"
	@echo "  build           Build CLI binary (dev mode, model from file)"
	@echo "  build-embedded  Build CLI with embedded model (-tags embed_model)"
	@echo "  deps            Download Go dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  bench           Run benchmarks"
	@echo ""
	@echo "Development:"
	@echo "  run-test        Run detection (INPUT=path OUTPUT=path)"
	@echo "  run-dev         Run detection with model path"
	@echo "  lint            Run golangci-lint"
	@echo "  fmt             Format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Remove build artifacts"
	@echo "  clean-all       Remove all artifacts including models"
	@echo ""
	@echo "Example workflow:"
	@echo "  make setup-python   # One-time Python setup"
	@echo "  make model          # Download and export model"
	@echo "  make build          # Build CLI (dev mode)"
	@echo "  make run-dev INPUT=image.png OUTPUT=result.png"
	@echo ""
	@echo "For production build with embedded model:"
	@echo "  make build-embedded # Build with go:embed model"
