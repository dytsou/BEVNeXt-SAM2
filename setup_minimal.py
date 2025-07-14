import os
import platform
import shutil
import sys
import warnings
from os import path as osp
from setuptools import find_packages, setup

import torch


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


# Merged requirements (Python 3.8 compatible)
REQUIRED_PACKAGES = [
    # Core dependencies - compatible versions for Python 3.8
    "torch>=1.13.0",
    "torchvision>=0.14.0",
    "numpy>=1.19.5",
    
    # SAM2 dependencies
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    
    # BEVNeXt dependencies
    "networkx>=2.2,<2.3",
    "numba>=0.53.0",
    "plyfile",
    "scikit-image",
    "tensorboard",
    "trimesh>=2.35.39,<2.35.40",
]

EXTRA_PACKAGES = {
    "all": REQUIRED_PACKAGES + [
        # Additional packages for full functionality
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "opencv-python>=4.7.0",
        "eva-decord>=0.6.1",
        "Flask>=3.0.3",
        "Flask-Cors>=5.0.0",
        "av>=13.0.0",
    ],
    "dev": [
        "black==24.2.0",
        "usort==1.0.2",
        "ufmt==2.0.0b2",
        "fvcore>=0.1.5.post20221221",
        "pandas>=2.2.2",
        "pycocotools>=2.0.8",
    ],
}


if __name__ == '__main__':
    # Minimal setup without CUDA extensions for Docker build
    setup(
        name='bevnext-sam2',
        version='1.0.0',
        description='BEVNeXt + SAM2: Unified 3D Object Detection and Segmentation',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='BEVNeXt-SAM2 Contributors',
        keywords='computer vision, 3D object detection, segmentation, BEV, SAM',
        url='https://github.com/your-repo/bevnext-sam2',
        packages=find_packages() + ['sam2'],
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
        ],
        python_requires='>=3.8.0',
        install_requires=REQUIRED_PACKAGES,
        extras_require=EXTRA_PACKAGES,
        # No ext_modules or cmdclass for minimal build
        zip_safe=False
    ) 