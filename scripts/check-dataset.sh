#!/bin/bash
# Check nuScenes dataset structure

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default path
DATA_PATH="${1:-../../nuscenes}"

echo -e "${BLUE}Checking nuScenes dataset structure...${NC}"
echo -e "${BLUE}Dataset path: $DATA_PATH${NC}"
echo ""

# Check if main directory exists
if [[ ! -d "$DATA_PATH" ]]; then
    echo -e "${RED}✗ Dataset directory not found: $DATA_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset directory exists${NC}"

# Check for v1.0-trainval directory
if [[ -d "$DATA_PATH/v1.0-trainval" ]]; then
    echo -e "${GREEN}✓ v1.0-trainval directory found${NC}"
    
    # Check key files
    REQUIRED_FILES=(
        "v1.0-trainval/attribute.json"
        "v1.0-trainval/calibrated_sensor.json"
        "v1.0-trainval/category.json"
        "v1.0-trainval/instance.json"
        "v1.0-trainval/log.json"
        "v1.0-trainval/map.json"
        "v1.0-trainval/sample.json"
        "v1.0-trainval/sample_annotation.json"
        "v1.0-trainval/sample_data.json"
        "v1.0-trainval/scene.json"
        "v1.0-trainval/sensor.json"
        "v1.0-trainval/visibility.json"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ -f "$DATA_PATH/$file" ]]; then
            echo -e "${GREEN}✓ $file${NC}"
        else
            echo -e "${RED}✗ Missing: $file${NC}"
        fi
    done
    
    # Check samples directory
    if [[ -d "$DATA_PATH/samples" ]]; then
        echo -e "${GREEN}✓ samples directory found${NC}"
        sample_count=$(find "$DATA_PATH/samples" -name "*.jpg" -o -name "*.png" | wc -l)
        echo -e "${GREEN}  └─ Found $sample_count image files${NC}"
    else
        echo -e "${RED}✗ samples directory missing${NC}"
    fi
    
    # Check sweeps directory
    if [[ -d "$DATA_PATH/sweeps" ]]; then
        echo -e "${GREEN}✓ sweeps directory found${NC}"
    else
        echo -e "${YELLOW}⚠ sweeps directory missing (optional)${NC}"
    fi
    
elif [[ -d "$DATA_PATH/v1.0-mini" ]]; then
    echo -e "${YELLOW}⚠ Found v1.0-mini instead of v1.0-trainval${NC}"
    echo -e "${YELLOW}  You can use the mini version for testing, but full training requires v1.0-trainval${NC}"
    
elif [[ -d "$DATA_PATH/v1.0-test" ]]; then
    echo -e "${YELLOW}⚠ Found v1.0-test instead of v1.0-trainval${NC}"
    echo -e "${YELLOW}  Test set cannot be used for training${NC}"
    
else
    echo -e "${RED}✗ No valid nuScenes version found${NC}"
    echo -e "${YELLOW}Available directories:${NC}"
    ls -la "$DATA_PATH"
    exit 1
fi

echo ""
echo -e "${BLUE}Dataset structure check completed!${NC}"

# Test with Python
echo ""
echo -e "${BLUE}Testing Python import...${NC}"

python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from nuscenes.nuscenes import NuScenes
    print('✓ nuScenes Python module available')
    
    # Try to load the dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='$DATA_PATH', verbose=False)
    print(f'✓ Dataset loaded successfully')
    print(f'  └─ Scenes: {len(nusc.scene)}')
    print(f'  └─ Samples: {len(nusc.sample)}')
    print(f'  └─ Sample annotations: {len(nusc.sample_annotation)}')
    
except ImportError as e:
    print(f'⚠ nuScenes module not available: {e}')
    print('  This is normal if running outside Docker container')
    
except Exception as e:
    print(f'✗ Failed to load dataset: {e}')
    sys.exit(1)
" 2>/dev/null || echo -e "${YELLOW}⚠ Python test failed (this is normal outside Docker)${NC}"

echo ""
echo -e "${GREEN}Dataset check completed!${NC}" 