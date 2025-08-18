# Simple Makefile helpers for BEVNeXt-SAM2

.PHONY: help build train-ddp monitor

DATA_PATH ?= /data/nuscenes
GPUS ?= 2
BATCH ?= 4
EPOCHS ?= 50
CONFIG ?=
RESUME ?=
DETACH ?=

CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
RESUME_FLAG := $(if $(RESUME),--resume $(RESUME),)
DETACH_FLAG := $(if $(DETACH),--detach,)

help:
	@echo "🚀 BEVNeXt-SAM2 Training with Optional Enhanced Checkpoint Resume"
	@echo ""
	@echo "Core Targets:"
	@echo "  build         Build Docker image (GPU-optimized)"
	@echo "  train-ddp     Launch multi-GPU DDP training in Docker"
	@echo "  monitor       Show container/logs and GPU status"
	@echo ""
	@echo "Basic Variables:"
	@echo "  DATA_PATH=/data/nuscenes  Path to nuScenes on host"
	@echo "  GPUS=2                    Number of GPUs or explicit IDs via GPUS_IDS"
	@echo "  GPUS_IDS=0,1              Optional explicit GPU IDs"
	@echo "  BATCH=4                   Per-GPU batch size"
	@echo "  EPOCHS=50                 Epochs"
	@echo "  CONFIG=path.json          Optional config file"
	@echo "  RESUME=path.pth           Optional checkpoint to resume"
	@echo "  DETACH=1                  Run container in background"
	@echo ""
	@echo "🔄 Optional Enhanced Features (add to train-ddp):"
	@echo "  AUTO_RESUME=1             Enable automatic resume detection"
	@echo "  CHECKPOINT_FREQ=50        Save checkpoint every N batches"
	@echo "  RESUME_FROM=path.pth      Resume from specific checkpoint"
	@echo "  NO_RESUME_PROMPT=1        Skip interactive resume prompts"
	@echo ""
	@echo "Examples:"
	@echo "  make train-ddp                                    # Standard training"
	@echo "  make train-ddp AUTO_RESUME=1                      # With auto-resume"
	@echo "  make train-ddp AUTO_RESUME=1 CHECKPOINT_FREQ=50   # Frequent checkpoints"
	@echo "  make train-ddp RESUME_FROM=outputs/best.pth       # Resume from specific"

build:
	./scripts/run.sh build-fast --gpu

# Enhanced train-ddp with optional checkpoint resume features
train-ddp:
	@echo "🚀 Starting enhanced multi-GPU training..."
	@echo "Enhanced features: $(if $(AUTO_RESUME),auto-resume,) $(if $(CHECKPOINT_FREQ),checkpoint-freq=$(CHECKPOINT_FREQ),) $(if $(RESUME_FROM),resume-from,) $(if $(NO_RESUME_PROMPT),no-prompts,)"
	./scripts/train_ddp_docker.sh \
		--data-path $(DATA_PATH) \
		$(if $(GPUS_IDS),--gpus $(GPUS_IDS),--num-gpus $(GPUS)) \
		--batch-size $(BATCH) \
		--epochs $(EPOCHS) \
		$(CONFIG_FLAG) \
		$(RESUME_FLAG) \
		$(DETACH_FLAG) \
		$(if $(AUTO_RESUME),--auto-resume,) \
		$(if $(CHECKPOINT_FREQ),--checkpoint-freq $(CHECKPOINT_FREQ),) \
		$(if $(RESUME_FROM),--resume-from $(RESUME_FROM),) \
		$(if $(NO_RESUME_PROMPT),--no-resume-prompt,)

monitor:
	./scripts/monitor_training.sh


