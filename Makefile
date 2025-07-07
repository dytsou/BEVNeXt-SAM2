# BEVNeXt-SAM2 Makefile
# Unified interface for all project operations

# Configuration
SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help setup build clean test lint format install install-dev \
	demo dev shell train-env inference-env tensorboard \
	compose-demo compose-dev compose-train compose-inference compose-tensorboard compose-down \
	train-bevnext train-sam2 train-fusion inference prepare-data create-gt-database update-coords \
	check-deps logs monitor-gpu status clean-data clean-logs clean-docker clean-all \
	quick-demo quick-dev quick-train benchmark profile convert-model advanced-inference \
	info debug rebuild

# Docker configuration
IMAGE_NAME := bevnext-sam2:latest
CONTAINER_NAME := bevnext-sam2-interactive

# Project paths
DATA_PATH := ./data
CHECKPOINTS_PATH := ./checkpoints
OUTPUTS_PATH := ./outputs
LOGS_PATH := ./logs

# GPU configuration
GPU_ID ?= 0
JUPYTER_PORT ?= 8888

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Common Docker run command for GPU operations
DOCKER_RUN_GPU := docker run --rm -it --runtime=nvidia \
	-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
	-v $(PWD)/data:/workspace/data \
	-v $(PWD)/checkpoints:/workspace/checkpoints \
	-v $(PWD)/outputs:/workspace/outputs \
	-v $(PWD)/logs:/workspace/logs \
	-v $(PWD):/workspace/bevnext-sam2 \
	-w /workspace/bevnext-sam2 \
	$(IMAGE_NAME)

# Common Docker run command for non-GPU operations
DOCKER_RUN := docker run --rm -it \
	-v $(PWD)/data:/workspace/data \
	-v $(PWD)/checkpoints:/workspace/checkpoints \
	-v $(PWD)/outputs:/workspace/outputs \
	-v $(PWD)/logs:/workspace/logs \
	-v $(PWD):/workspace/bevnext-sam2 \
	-w /workspace/bevnext-sam2 \
	$(IMAGE_NAME)

##@ Help
help: ## Display this help
	@echo -e "$(BLUE)BEVNeXt-SAM2 Makefile (Docker-Only)$(NC)"
	@echo -e "$(BLUE)====================================$(NC)"
	@echo -e "$(YELLOW)⚠ ALL operations run in Docker containers$(NC)"
	@echo -e "$(YELLOW)⚠ Docker and NVIDIA Docker runtime required$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup and Installation
setup: ## Setup project directories and check prerequisites
	@echo -e "$(BLUE)Setting up BEVNeXt-SAM2 project...$(NC)"
	@if ! docker info >/dev/null 2>&1; then \
		echo -e "$(RED)✗ Docker is not running or not accessible$(NC)"; \
		echo -e "$(YELLOW)Please start Docker and try again$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✓ Docker is available$(NC)"
	@if docker info 2>/dev/null | grep -q nvidia; then \
		echo -e "$(GREEN)✓ NVIDIA Docker runtime detected$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ NVIDIA Docker runtime not detected - GPU operations may not work$(NC)"; \
	fi
	@mkdir -p $(DATA_PATH) $(CHECKPOINTS_PATH) $(OUTPUTS_PATH) $(LOGS_PATH)
	@chmod +x scripts/*.sh
	@echo -e "$(GREEN)✓ Project directories created$(NC)"
	@echo -e "$(GREEN)✓ Scripts made executable$(NC)"
	@echo -e "$(GREEN)Setup complete!$(NC)"

install: build ## Install dependencies (Docker-based)
	@echo -e "$(BLUE)Installing BEVNeXt-SAM2 dependencies in Docker...$(NC)"
	docker run --rm -it $(IMAGE_NAME) pip install -e .
	@echo -e "$(GREEN)✓ Installation complete!$(NC)"

install-dev: build ## Install development dependencies (Docker-based)
	@echo -e "$(BLUE)Installing development dependencies in Docker...$(NC)"
	docker run --rm -it $(IMAGE_NAME) pip install -e ".[dev]"
	@echo -e "$(GREEN)✓ Development installation complete!$(NC)"

install-all: build ## Install all optional dependencies (Docker-based)
	@echo -e "$(BLUE)Installing all dependencies in Docker...$(NC)"
	docker run --rm -it $(IMAGE_NAME) pip install -e ".[all]"
	@echo -e "$(GREEN)✓ Full installation complete!$(NC)"

##@ Docker Operations
build: setup ## Build Docker image
	@echo -e "$(BLUE)Building Docker image...$(NC)"
	./scripts/build.sh
	@echo -e "$(GREEN)✓ Docker build complete!$(NC)"

rebuild: clean-docker build ## Clean and rebuild Docker image
	@echo -e "$(GREEN)✓ Rebuild complete!$(NC)"

##@ Running Modes
demo: ## Run the fusion demo
	@echo -e "$(BLUE)Running BEVNeXt-SAM2 demo...$(NC)"
	./scripts/run.sh demo --gpu $(GPU_ID)

dev: ## Start development environment with Jupyter
	@echo -e "$(BLUE)Starting development environment...$(NC)"
	@echo -e "$(YELLOW)Jupyter will be available at: http://localhost:$(JUPYTER_PORT)$(NC)"
	./scripts/run.sh dev --port $(JUPYTER_PORT) --gpu $(GPU_ID)

shell: ## Start interactive shell in container
	@echo -e "$(BLUE)Starting interactive shell...$(NC)"
	./scripts/run.sh shell --gpu $(GPU_ID)

train-env: ## Start training environment
	@echo -e "$(BLUE)Starting training environment...$(NC)"
	./scripts/run.sh train --gpu $(GPU_ID)

inference-env: ## Start inference environment
	@echo -e "$(BLUE)Starting inference environment...$(NC)"
	./scripts/run.sh inference --gpu $(GPU_ID)

tensorboard: ## Start TensorBoard server
	@echo -e "$(BLUE)Starting TensorBoard...$(NC)"
	@echo -e "$(YELLOW)TensorBoard will be available at: http://localhost:6006$(NC)"
	./scripts/run.sh tensorboard

##@ Docker Compose Operations
compose-demo: ## Run demo using docker-compose
	@echo -e "$(BLUE)Running demo with docker-compose...$(NC)"
	docker-compose up --build bevnext-sam2

compose-dev: ## Start development environment using docker-compose
	@echo -e "$(BLUE)Starting development environment with docker-compose...$(NC)"
	docker-compose up --build dev

compose-train: ## Start training environment using docker-compose
	@echo -e "$(BLUE)Starting training environment with docker-compose...$(NC)"
	docker-compose up --build train

compose-inference: ## Start inference environment using docker-compose
	@echo -e "$(BLUE)Starting inference environment with docker-compose...$(NC)"
	docker-compose up --build inference

compose-tensorboard: ## Start TensorBoard using docker-compose
	@echo -e "$(BLUE)Starting TensorBoard with docker-compose...$(NC)"
	docker-compose up --build tensorboard

compose-down: ## Stop all docker-compose services
	@echo -e "$(BLUE)Stopping all docker-compose services...$(NC)"
	docker-compose down

##@ Training Operations
train-bevnext: ## Train BEVNeXt model (Docker)
	@echo -e "$(BLUE)Training BEVNeXt model in Docker...$(NC)"
	$(DOCKER_RUN_GPU) python tools/train.py configs/bevnext/bevnext-stage1.py

train-sam2: ## Train SAM2 model (Docker)
	@echo -e "$(BLUE)Training SAM2 model in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD)/logs:/workspace/logs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/train.py configs/sam2/sam2_finetune.yaml

train-fusion: ## Train fusion model (Docker)
	@echo -e "$(BLUE)Training fusion model in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD)/logs:/workspace/logs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/train.py configs/fusion_config.yaml

##@ Testing and Inference
test: ## Run tests (Docker)
	@echo -e "$(BLUE)Running tests in Docker...$(NC)"
	$(DOCKER_RUN_GPU) python tools/test.py configs/bevnext/bevnext-stage1.py checkpoints/bevnext_checkpoint.pth

inference: ## Run inference (Docker)
	@echo -e "$(BLUE)Running inference in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/test.py \
		configs/bevnext/bevnext-stage1.py \
		checkpoints/bevnext_checkpoint.pth \
		--out outputs/inference_results.pkl \
		--eval bbox

##@ Data Management
prepare-data: ## Prepare dataset (Docker)
	@echo -e "$(BLUE)Preparing dataset in Docker...$(NC)"
	$(DOCKER_RUN) python tools/create_data.py

create-gt-database: ## Create ground truth database (Docker)
	@echo -e "$(BLUE)Creating ground truth database in Docker...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/data_converter/create_gt_database.py

update-coords: ## Update data coordinates (Docker)
	@echo -e "$(BLUE)Updating data coordinates in Docker...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		bash tools/update_data_coords.sh

##@ Development Tools
lint: ## Run linting (Docker)
	@echo -e "$(BLUE)Running linting in Docker...$(NC)"
	docker run --rm -it \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		black --check .

format: ## Format code (Docker)
	@echo -e "$(BLUE)Formatting code in Docker...$(NC)"
	docker run --rm -it \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		black .
	@echo -e "$(GREEN)✓ Code formatted!$(NC)"

check-deps: ## Check dependencies (Docker)
	@echo -e "$(BLUE)Checking dependencies in Docker...$(NC)"
	@docker run --rm $(IMAGE_NAME) python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo -e "$(RED)✗ PyTorch not installed$(NC)"
	@docker run --rm $(IMAGE_NAME) python -c "import mmcv; print(f'✓ MMCV: {mmcv.__version__}')" 2>/dev/null || echo -e "$(RED)✗ MMCV not installed$(NC)"
	@docker run --rm $(IMAGE_NAME) python -c "import mmdet; print(f'✓ MMDet: {mmdet.__version__}')" 2>/dev/null || echo -e "$(RED)✗ MMDet not installed$(NC)"

##@ Monitoring and Logs
logs: ## View Docker logs
	@echo -e "$(BLUE)Viewing recent logs...$(NC)"
	@if [ -d "$(LOGS_PATH)" ] && [ -n "$$(ls -A $(LOGS_PATH) 2>/dev/null)" ]; then \
		find $(LOGS_PATH) -name "*.log" -type f -exec tail -f {} +; \
	else \
		echo -e "$(YELLOW)No log files found in $(LOGS_PATH)$(NC)"; \
	fi

monitor-gpu: ## Monitor GPU usage
	@echo -e "$(BLUE)Monitoring GPU usage...$(NC)"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		watch -n 1 nvidia-smi; \
	else \
		echo -e "$(RED)nvidia-smi not available$(NC)"; \
	fi

status: ## Show system status
	@echo -e "$(BLUE)System Status$(NC)"
	@echo -e "$(BLUE)==============$(NC)"
	@echo "Docker Status:"
	@if docker info >/dev/null 2>&1; then \
		echo -e "$(GREEN)✓ Docker is running$(NC)"; \
		if docker info 2>/dev/null | grep -q nvidia; then \
			echo -e "$(GREEN)✓ NVIDIA Docker runtime available$(NC)"; \
		else \
			echo -e "$(YELLOW)⚠ NVIDIA Docker runtime not detected$(NC)"; \
		fi \
	else \
		echo -e "$(RED)✗ Docker is not running$(NC)"; \
	fi
	@echo ""
	@echo "GPU Status:"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
		awk -F', ' '{printf "  GPU: %s | Memory: %s/%s MB | Utilization: %s%%\n", $$1, $$2, $$3, $$4}'; \
	else \
		echo -e "$(YELLOW)  No NVIDIA GPUs detected$(NC)"; \
	fi

##@ Cleanup
clean: ## Clean build artifacts
	@echo -e "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.so" -delete
	@echo -e "$(GREEN)✓ Build artifacts cleaned$(NC)"

clean-data: ## Clean generated data (use with caution)
	@echo -e "$(YELLOW)Warning: This will delete generated data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf $(OUTPUTS_PATH)/*
	@echo -e "$(GREEN)✓ Generated data cleaned$(NC)"

clean-logs: ## Clean log files
	@echo -e "$(BLUE)Cleaning log files...$(NC)"
	rm -rf $(LOGS_PATH)/*.log
	@echo -e "$(GREEN)✓ Log files cleaned$(NC)"

clean-docker: ## Clean Docker artifacts
	@echo -e "$(BLUE)Cleaning Docker artifacts...$(NC)"
	docker system prune -f
	@if docker images $(IMAGE_NAME) -q | head -n 1; then \
		docker rmi $(IMAGE_NAME) || true; \
	fi
	@echo -e "$(GREEN)✓ Docker artifacts cleaned$(NC)"

clean-all: clean clean-logs clean-docker ## Clean everything
	@echo -e "$(GREEN)✓ Complete cleanup finished$(NC)"

##@ Quick Actions
quick-demo: build demo ## Build Docker image and run demo
	@echo -e "$(GREEN)✓ Quick demo complete!$(NC)"

quick-dev: build dev ## Build Docker image and start development environment
	@echo -e "$(GREEN)✓ Quick development setup complete!$(NC)"

quick-train: build train-bevnext ## Build Docker image and start training
	@echo -e "$(GREEN)✓ Quick training started!$(NC)"

##@ Advanced Operations
benchmark: ## Run benchmarks (Docker)
	@echo -e "$(BLUE)Running benchmarks in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/analysis_tools/benchmark.py \
		configs/bevnext/bevnext-stage1.py \
		checkpoints/bevnext_checkpoint.pth

profile: ## Profile the model (Docker)
	@echo -e "$(BLUE)Profiling model in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/analysis_tools/get_flops.py \
		configs/bevnext/bevnext-stage1.py

convert-model: ## Convert model format (Docker)
	@echo -e "$(BLUE)Converting model format in Docker...$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/convert_bevdet_to_TRT.py \
		configs/bevnext/bevnext-stage1.py \
		checkpoints/bevnext_checkpoint.pth \
		--work-dir outputs/converted_models

# Variables for advanced users
ADVANCED_CONFIG ?= configs/bevnext/bevnext-stage1.py
ADVANCED_CHECKPOINT ?= checkpoints/bevnext_checkpoint.pth
ADVANCED_OUTPUT ?= outputs/advanced_results.pkl

advanced-inference: ## Advanced inference with custom parameters (Docker)
	@echo -e "$(BLUE)Running advanced inference in Docker...$(NC)"
	@echo -e "$(BLUE)Config: $(ADVANCED_CONFIG)$(NC)"
	@echo -e "$(BLUE)Checkpoint: $(ADVANCED_CHECKPOINT)$(NC)"
	@echo -e "$(BLUE)Output: $(ADVANCED_OUTPUT)$(NC)"
	docker run --rm -it --runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/outputs:/workspace/outputs \
		-v $(PWD):/workspace/bevnext-sam2 \
		-w /workspace/bevnext-sam2 \
		$(IMAGE_NAME) \
		python tools/test.py \
		$(ADVANCED_CONFIG) \
		$(ADVANCED_CHECKPOINT) \
		--out $(ADVANCED_OUTPUT) \
		--eval bbox

##@ Information
info: ## Show project information
	@echo -e "$(BLUE)BEVNeXt-SAM2 Project Information$(NC)"
	@echo -e "$(BLUE)================================$(NC)"
	@echo "Version: 1.0.0"
	@echo "Description: Unified 3D Object Detection and Segmentation"
	@echo "Components: BEVNeXt + SAM 2"
	@echo ""
	@echo "Key Directories:"
	@echo "  Data: $(DATA_PATH)"
	@echo "  Checkpoints: $(CHECKPOINTS_PATH)"
	@echo "  Outputs: $(OUTPUTS_PATH)"
	@echo "  Logs: $(LOGS_PATH)"
	@echo ""
	@echo "Docker Image: $(IMAGE_NAME)"
	@echo "Default GPU: $(GPU_ID)"
	@echo "Jupyter Port: $(JUPYTER_PORT)"

# Debug target for troubleshooting
debug: ## Debug common issues
	@echo -e "$(BLUE)Debug Information$(NC)"
	@echo -e "$(BLUE)=================$(NC)"
	@echo "Current directory: $(PWD)"
	@echo "Shell: $(SHELL)"
	@echo "Make version: $(MAKE_VERSION)"
	@echo ""
	@echo "Directory permissions:"
	@ls -la scripts/ | head -5
	@echo ""
	@echo "Docker version:"
	@docker --version 2>/dev/null || echo "Docker not available"
	@echo ""
	@echo "Python version:"
	@python --version 2>/dev/null || echo "Python not available"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_VISIBLE_DEVICES: $${CUDA_VISIBLE_DEVICES:-not set}"
	@echo "  GPU_ID: $(GPU_ID)"