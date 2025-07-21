#!/usr/bin/env python3
"""
Install nuScenes dependencies for BEVNeXt-SAM2 training

This script installs the required packages for using nuScenes dataset.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages"""
    print("Installing nuScenes dependencies...")
    
    packages = [
        "nuscenes-devkit",  # Official nuScenes devkit
        "pyquaternion",     # For quaternion operations
        "shapely",          # For geometric operations
        "descartes",        # For polygon plotting
        "matplotlib>=3.0",  # For visualization
        "pillow>=8.0",      # For image operations
        "opencv-python",    # For computer vision
    ]
    
    success_count = 0
    for package in packages:
        print(f"   Installing {package}...")
        if install_package(package):
            print(f"   {package} installed successfully")
            success_count += 1
        else:
            print(f"   Failed to install {package}")
    
    print(f"\n📊 Installation summary: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("All dependencies installed successfully!")
        print("\nTo use nuScenes dataset, ensure your data is structured as:")
        print("   data/nuscenes/")
        print("   ├── samples/")
        print("   ├── sweeps/") 
        print("   ├── maps/")
        print("   └── v1.0-trainval/ (or v1.0-mini/)")
        print("       ├── sample.json")
        print("       ├── sample_data.json")
        print("       └── ... (other JSON files)")
    else:
        print("Some dependencies failed to install. Training may fall back to synthetic data.")
    
    return success_count == len(packages)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 