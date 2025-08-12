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
	@echo "Targets:"
	@echo "  build         Build Docker image (GPU-optimized)"
	@echo "  train-ddp     Launch 2x GPU DDP training in Docker"
	@echo "  monitor       Show container/logs and GPU status"
	@echo ""
	@echo "Variables:"
	@echo "  DATA_PATH=/data/nuscenes  Path to nuScenes on host"
	@echo "  GPUS=2                    Number of GPUs or explicit IDs via GPUS_IDS"
	@echo "  GPUS_IDS=0,1              Optional explicit GPU IDs"
	@echo "  BATCH=4                   Per-GPU batch size"
	@echo "  EPOCHS=50                 Epochs"
	@echo "  CONFIG=path.json          Optional config file"
	@echo "  RESUME=path.pth           Optional checkpoint to resume"
	@echo "  DETACH=1                  Run container in background"

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


