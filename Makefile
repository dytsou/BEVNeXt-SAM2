# Simple Makefile helpers for BEVNeXt-SAM2

.PHONY: help build train-ddp monitor validate validate-quick validate-full validate-dataset

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
OUTPUT_DIR ?= outputs/validation
MAX_SAMPLES ?= 100
DEVICE ?= auto
DOCKER ?=

CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
RESUME_FLAG := $(if $(RESUME),--resume $(RESUME),)
DETACH_FLAG := $(if $(DETACH),--detach,)
DOCKER_FLAG := $(if $(DOCKER),--docker,)

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
	@echo "Validation Variables:"
	@echo "  CHECKPOINT=checkpoints/latest.pth  Model checkpoint to validate"
	@echo "  OUTPUT_DIR=outputs/validation      Validation output directory"
	@echo "  MAX_SAMPLES=100                    Max samples for validation"
	@echo "  DEVICE=auto                        Device: auto, cuda, cpu"
	@echo "  DOCKER=1                           Use Docker for validation"
	@echo ""
	@echo "Examples:"
	@echo "  make validate DATA_PATH=/data/nuscenes CHECKPOINT=models/best.pth"
	@echo "  make validate-full DOCKER=1"
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

validate-quick:
	@echo "🔍 Running quick model validation..."
	python quick_validate.py \
		--checkpoint $(CHECKPOINT) \
		--data-root $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--max-samples $(MAX_SAMPLES) \
		--device $(DEVICE) \
		$(DOCKER_FLAG)

validate-full:
	@echo "🔍 Running full model validation with all metrics..."
	python quick_validate.py \
		--checkpoint $(CHECKPOINT) \
		--data-root $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--device $(DEVICE) \
		--full \
		$(DOCKER_FLAG)

validate-dataset:
	@echo "🔍 Running dataset integrity validation..."
	python quick_validate.py \
		--dataset-only \
		--data-root $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		$(DOCKER_FLAG)

# Alternative shell script validation (for CI/CD)
validate-shell:
	@echo "🔍 Running validation with shell script..."
	./scripts/quick_validate.sh \
		--checkpoint $(CHECKPOINT) \
		--data-root $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--max-samples $(MAX_SAMPLES) \
		--device $(DEVICE) \
		$(DOCKER_FLAG)

validate-shell-full:
	@echo "🔍 Running full validation with shell script..."
	./scripts/quick_validate.sh \
		--checkpoint $(CHECKPOINT) \
		--data-root $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--device $(DEVICE) \
		--full \
		$(DOCKER_FLAG)


