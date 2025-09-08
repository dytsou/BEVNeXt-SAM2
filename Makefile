# Simple Makefile helpers for BEVNeXt-SAM2

.PHONY: help build train-ddp monitor init-dirs clean-checkpoints status

DATA_PATH ?= /data/nuscenes
GPUS ?= 2
BATCH ?= 4
EPOCHS ?= 50
CONFIG ?=
RESUME ?=
DETACH ?=

# Checkpoint and output directories
OUTPUTS_DIR := outputs
CHECKPOINTS_DIR := $(OUTPUTS_DIR)/checkpoints
TRAINING_DIR := $(OUTPUTS_DIR)/training
EVAL_DIR := $(OUTPUTS_DIR)/evaluation
LOGS_DIR := logs

CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
RESUME_FLAG := $(if $(RESUME),--resume $(RESUME),)
DETACH_FLAG := $(if $(DETACH),--detach,)

help:
	@echo "🚀 BEVNeXt-SAM2 Training with Enhanced Docker Volume Management"
	@echo ""
	@echo "Core Targets:"
	@echo "  build         Build Docker image (GPU-optimized)"
	@echo "  init-dirs     Initialize output and checkpoint directories"
	@echo "  train-ddp     Launch multi-GPU DDP training in Docker"
	@echo "  monitor       Show container/logs and GPU status"
	@echo "  status        Show current project and checkpoint status"
	@echo "  clean-checkpoints  Clean old checkpoint files"
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
	@echo "🔄 Enhanced Checkpoint Features (add to train-ddp):"
	@echo "  AUTO_RESUME=1             Enable automatic resume detection"
	@echo "  CHECKPOINT_FREQ=50        Save checkpoint every N batches"
	@echo "  RESUME_FROM=path.pth      Resume from specific checkpoint"
	@echo "  NO_RESUME_PROMPT=1        Skip interactive resume prompts"
	@echo ""
	@echo "📁 Directory Structure:"
	@echo "  $(CHECKPOINTS_DIR)/       Enhanced checkpoint storage"
	@echo "  $(TRAINING_DIR)/          Training outputs and logs"
	@echo "  $(EVAL_DIR)/              Evaluation results"
	@echo "  $(LOGS_DIR)/              Application logs"
	@echo ""
	@echo "Examples:"
	@echo "  make init-dirs                                        # Initialize directories"
	@echo "  make train-ddp                                        # Standard training"
	@echo "  make train-ddp AUTO_RESUME=1                          # With auto-resume"
	@echo "  make train-ddp AUTO_RESUME=1 CHECKPOINT_FREQ=50       # Frequent checkpoints"
	@echo "  make train-ddp RESUME_FROM=$(CHECKPOINTS_DIR)/best.pth # Resume from specific"

# Initialize required directories with proper permissions
init-dirs:
	@echo "📁 Initializing Docker volume mounts..."
	./scripts/init_docker_volumes.sh

build: init-dirs
	@echo "🏗️ Building Docker image with directory initialization..."
	./scripts/run.sh build-fast --gpu
	@echo "✅ Build complete with proper volume mount structure"

# Enhanced train-ddp with proper volume mounts and checkpoint management
train-ddp: init-dirs
	@echo "🚀 Starting enhanced multi-GPU training with proper volume mounts..."
	@echo "📁 Checkpoint directory: $(CHECKPOINTS_DIR)/"
	@echo "📊 Training outputs: $(TRAINING_DIR)/"
	@echo "🔄 Enhanced features: $(if $(AUTO_RESUME),auto-resume,) $(if $(CHECKPOINT_FREQ),checkpoint-freq=$(CHECKPOINT_FREQ),) $(if $(RESUME_FROM),resume-from,) $(if $(NO_RESUME_PROMPT),no-prompts,)"
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
	@echo "🔍 After training, checkpoints should be in: $(CHECKPOINTS_DIR)/"

# Enhanced monitoring with checkpoint status
monitor:
	@echo "📊 Training and Checkpoint Status"
	@echo "================================"
	./scripts/monitor_training.sh
	@echo ""
	@echo "📁 Current checkpoint files:"
	@find $(CHECKPOINTS_DIR) -name "checkpoint_*.pth" -type f 2>/dev/null | head -10 || echo "   No checkpoint files found in $(CHECKPOINTS_DIR)/"
	@echo ""
	@echo "📈 Training directory contents:"
	@ls -la $(TRAINING_DIR)/ 2>/dev/null || echo "   Training directory not found"

# Show current checkpoint and volume status
status:
	@echo "📊 BEVNeXt-SAM2 Project Status"
	@echo "============================="
	@echo ""
	@echo "📁 Directory Structure:"
	@echo "  Checkpoints: $(CHECKPOINTS_DIR)/"
	@echo "  Training: $(TRAINING_DIR)/"
	@echo "  Evaluation: $(EVAL_DIR)/"
	@echo "  Logs: $(LOGS_DIR)/"
	@echo ""
	@echo "💾 Disk Usage:"
	@du -sh $(OUTPUTS_DIR)/* 2>/dev/null || echo "   No output directories found"
	@echo ""
	@echo "🔄 Latest Checkpoints:"
	@find $(CHECKPOINTS_DIR) -name "checkpoint_*.pth" -type f -exec ls -lh {} \; 2>/dev/null | tail -5 || echo "   No checkpoints found"
	@echo ""
	@echo "🐳 Docker Containers:"
	@docker ps -a --filter "name=bevnext" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "   No Docker containers found"

# Clean old checkpoint files (keep latest and best)
clean-checkpoints:
	@echo "🧹 Cleaning old checkpoint files..."
	@echo "Keeping latest and best checkpoints"
	@find $(CHECKPOINTS_DIR) -name "checkpoint_epoch_*.pth" -type f | sort | head -n -5 | xargs rm -f 2>/dev/null || true
	@echo "✅ Cleanup complete"
	@ls -la $(CHECKPOINTS_DIR)/ 2>/dev/null || echo "   No checkpoints found"