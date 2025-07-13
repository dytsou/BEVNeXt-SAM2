# BEVNeXt-SAM2 Cleanup Plan

## Overview
This document outlines the cleanup strategy for the BEVNeXt-SAM2 project to focus on training and testing functionality.

## Files Analysis

### ✅ Core Files to KEEP
**Training & Testing**
- `tools/train.py` - Standard mmdet3d training
- `tools/test.py` - Standard mmdet3d testing  
- `training/train_bevnext_sam2.py` - Specialized BEVNeXt-SAM2 training with synthetic data
- `training/config_*.json` - Training configurations

**Integration & Models**
- `integration/bev_sam_fusion.py` - BEV-SAM fusion implementation
- `integration/sam_enhanced_detector.py` - Enhanced detector
- `bevnext/` - BEVNeXt module (core)
- `sam2_module/` - SAM2 module (core)

**Docker & Scripts**
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration
- `scripts/run.sh` - Docker runner script
- `scripts/build.sh` - Build script

**Configuration & Setup**
- `setup.py` - Package setup
- `pyproject.toml` - Project metadata
- `constraints.txt` - Dependency constraints
- `configs/` - Configuration files

**Documentation**
- `README.md` - Main documentation
- `DOCKER.md` - Docker documentation
- `DEPENDENCY_FIX_COMPLETE.md` - Dependency fixes

### ❌ Files to REMOVE (Non-essential for training/testing)

**Analysis Tools** (Can be removed - only used for advanced analysis)
- `tools/analysis_tools/benchmark.py`
- `tools/analysis_tools/benchmark_sequential.py`
- `tools/analysis_tools/benchmark_trt.py`
- `tools/analysis_tools/benchmark_view_transformer.py`
- `tools/analysis_tools/get_flops.py`
- `tools/analysis_tools/vis.py`

**Model Converters** (Can be removed - not used by BEVNeXt-SAM2)
- `tools/model_converters/convert_votenet_checkpoints.py`
- `tools/model_converters/convert_h3dnet_checkpoints.py`
- `tools/model_converters/regnet2mmdet.py`
- `tools/model_converters/publish_model.py`

**Data Converters** (Can be removed - using synthetic data)
- `tools/data_converter/kitti_converter.py`
- `tools/data_converter/nuscenes_converter.py`
- `tools/data_converter/waymo_converter.py`
- `tools/data_converter/lyft_converter.py`
- `tools/data_converter/indoor_converter.py`
- `tools/data_converter/nuimage_converter.py`
- `tools/data_converter/scannet_data_utils.py`
- `tools/data_converter/sunrgbd_data_utils.py`
- `tools/data_converter/s3dis_data_utils.py`
- `tools/data_converter/lyft_data_fixer.py`

**Unused Examples**
- ✅ `examples/demo_bev_simple.py` - Already removed (empty file)

**Unused Configs** (Keep essential ones only)
- Many configs in `configs/bevnext/` for unsupported models

## Cleanup Strategy

### Phase 1: Remove Analysis Tools (Low Risk)
These are only used for advanced analysis and not needed for basic training/testing.

### Phase 2: Remove Model Converters (Medium Risk)
These are only used for specific model types not used in BEVNeXt-SAM2.

### Phase 3: Remove Data Converters (Medium Risk)
BEVNeXt-SAM2 uses synthetic data, so external dataset converters are not needed.

### Phase 4: Config Cleanup (High Risk)
Carefully review and remove only configs for unsupported models.

## Impact Assessment

### File Count Reduction
- **Before**: 501 Python files
- **After**: ~400 Python files (estimated)
- **Reduction**: ~100 files (~20%)

### Functionality Impact
- **Training**: ✅ No impact (uses synthetic data)
- **Testing**: ✅ No impact (core functionality preserved)
- **Docker**: ✅ No impact (core Docker files kept)
- **Integration**: ✅ No impact (core integration files kept)

## Validation Plan

### After Each Phase
1. **Docker Build Test**: Ensure Docker image builds successfully
2. **Training Test**: Run synthetic data training for 1 epoch
3. **Testing Test**: Run inference/testing pipeline
4. **Integration Test**: Verify BEV-SAM fusion works

### Final Validation
1. **Complete Training Run**: Train for multiple epochs
2. **Performance Test**: Validate training metrics
3. **Memory Test**: Check GPU memory usage
4. **Documentation Update**: Update README with changes

## Risk Mitigation

### Low Risk Actions
- Create backup of original codebase
- Remove files in phases with validation
- Test after each phase

### High Risk Actions
- Config file removal (requires careful analysis)
- Core module changes (avoid unless necessary)

## Expected Benefits

### Immediate
- **Reduced complexity**: Easier to navigate codebase
- **Faster builds**: Less files to process
- **Cleaner structure**: Focus on essential functionality

### Long-term
- **Easier maintenance**: Less code to maintain
- **Better performance**: Reduced overhead
- **Clearer documentation**: Focus on core features

## Next Steps

1. **Execute Phase 1**: Remove analysis tools
2. **Validate**: Test Docker build and training
3. **Execute Phase 2**: Remove model converters
4. **Validate**: Test Docker build and training
5. **Execute Phase 3**: Remove data converters
6. **Validate**: Test Docker build and training
7. **Final test**: Complete training and testing run

---
*Cleanup plan created for BEVNeXt-SAM2 optimization*