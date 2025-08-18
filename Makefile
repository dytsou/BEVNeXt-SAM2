# Simple Makefile helpers for BEVNeXt-SAM2

.PHONY: help build train-ddp monitor validate validate-quick validate-full validate-dataset ensure-docker

# Training variables
DATA_PATH ?= /data/nuscenes
GPUS ?= 2
BATCH ?= 4
EPOCHS ?= 50
CONFIG ?=
RESUME ?=
DETACH ?=

# Validation variables  
CHECKPOINT ?= checkpoints/latest.pth
GPU_SUPPORT ?= 1

CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
RESUME_FLAG := $(if $(RESUME),--resume $(RESUME),)
DETACH_FLAG := $(if $(DETACH),--detach,)


help:
	@echo "🚀 BEVNeXt-SAM2 Makefile Commands"
	@echo ""
	@echo "Training Targets:"
	@echo "  build         Build Docker image (GPU-optimized)"
	@echo "  train-ddp     Launch multi-GPU DDP training in Docker"
	@echo "  monitor       Show container/logs and GPU status"
	@echo ""
	@echo "Validation Targets:"
	@echo "  validate         Quick model validation (recommended)"
	@echo "  validate-quick   Fast validation with minimal samples"
	@echo "  validate-full    Complete validation with all metrics + visualizations"
	@echo "  validate-dataset Dataset integrity check only"
	@echo ""
	@echo "Training Variables:"
	@echo "  DATA_PATH=/data/nuscenes  Path to nuScenes dataset"
	@echo "  GPUS=2                    Number of GPUs (or use GPUS_IDS=0,1)"
	@echo "  BATCH=4                   Per-GPU batch size"
	@echo "  EPOCHS=50                 Training epochs"
	@echo "  CONFIG=config.json        Training configuration file"
	@echo "  RESUME=checkpoint.pth     Checkpoint to resume from"
	@echo "  DETACH=1                  Run container in background"
	@echo ""
	@echo "Validation Variables (Docker-based):"
	@echo "  CHECKPOINT=checkpoints/latest.pth  Model checkpoint to validate"
	@echo "  GPU_SUPPORT=1                      Enable GPU support (0 to disable)"
	@echo "  Note: All validation runs in Docker automatically"
	@echo ""
	@echo "Examples:"
	@echo "  make validate DATA_PATH=/data/nuscenes CHECKPOINT=models/best.pth"
	@echo "  make validate-full GPU_SUPPORT=0  # CPU-only validation"
	@echo "  make validate-dataset"
	@echo "  make train-ddp DATA_PATH=/data/nuscenes GPUS=4 BATCH=2"

build:
	./scripts/run.sh build-fast --gpu

train-ddp:
	./scripts/train_ddp_docker.sh \
		--data-path $(DATA_PATH) \
		$(if $(GPUS_IDS),--gpus $(GPUS_IDS),--num-gpus $(GPUS)) \
		--batch-size $(BATCH) \
		--epochs $(EPOCHS) \
		$(CONFIG_FLAG) \
		$(RESUME_FLAG) \
		$(DETACH_FLAG)

monitor:
	./scripts/monitor_training.sh

# Validation targets
validate: validate-quick
	@echo "✅ Quick validation completed. Use 'make validate-full' for comprehensive validation."



# Add dependency to auto-build Docker image for all validation targets
.PHONY: ensure-docker
ensure-docker:
	@if ! docker image inspect bevnext-sam2 >/dev/null 2>&1; then \
		echo "🔨 Docker image not found. Building bevnext-sam2 image..."; \
		$(MAKE) build; \
	else \
		echo "✅ Docker image bevnext-sam2 found"; \
	fi


# Override validate targets to use ensure-docker
validate-quick: ensure-docker
	@echo "🔍 Running quick model validation with Docker..."
	@if [ "$(GPU_SUPPORT)" = "1" ]; then \
		echo "🐳 Using Docker for validation with GPU support"; \
		./scripts/run.sh validate \
			--checkpoint $(CHECKPOINT) \
			--data-path $(DATA_PATH) \
			--gpu; \
	else \
		echo "🐳 Using Docker for validation (CPU only)"; \
		./scripts/run.sh validate \
			--checkpoint $(CHECKPOINT) \
			--data-path $(DATA_PATH) \
			--no-gpu; \
	fi

validate-full: ensure-docker
	@echo "🔍 Running full model validation with all metrics..."
	@if [ "$(GPU_SUPPORT)" = "1" ]; then \
		echo "🐳 Using Docker for full validation with GPU support"; \
		./scripts/run.sh validate \
			--checkpoint $(CHECKPOINT) \
			--data-path $(DATA_PATH) \
			--gpu; \
	else \
		echo "🐳 Using Docker for full validation (CPU only)"; \
		./scripts/run.sh validate \
			--checkpoint $(CHECKPOINT) \
			--data-path $(DATA_PATH) \
			--no-gpu; \
	fi

validate-dataset: ensure-docker
	@echo "🔍 Running dataset integrity validation..."
	@if [ "$(GPU_SUPPORT)" = "1" ]; then \
		echo "🐳 Using Docker for dataset validation with GPU support"; \
		./scripts/run.sh validate-nuscenes \
			--data-path $(DATA_PATH) \
			--gpu; \
	else \
		echo "🐳 Using Docker for dataset validation (CPU only)"; \
		./scripts/run.sh validate-nuscenes \
			--data-path $(DATA_PATH) \
			--no-gpu; \
	fi


