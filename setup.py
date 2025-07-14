import os
import platform
import shutil
import sys
import warnings
from os import path as osp
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    force_cuda = os.getenv('FORCE_CUDA', '0')
    if force_cuda == '0':
        # Explicitly disable CUDA compilation
        print('Compiling {} without CUDA (FORCE_CUDA=0)'.format(name))
        extension = CppExtension
    elif torch.cuda.is_available() or force_cuda == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def get_sam2_extensions():
    """Get SAM2 CUDA extensions"""
    force_cuda = os.getenv('FORCE_CUDA', '0')
    if force_cuda == '0' or (not torch.cuda.is_available() and force_cuda != '1'):
        print("Skipping SAM2 CUDA extensions (FORCE_CUDA=0 or CUDA not available)")
        return []
    
    try:
        from torch.utils.cpp_extension import CUDAExtension
        
        # Ensure sam2 directory exists
        os.makedirs('sam2', exist_ok=True)
        
        srcs = ["sam2_module/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        return [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        print(f"Failed to build SAM 2 CUDA extension: {e}")
        return []


# Merged requirements
REQUIRED_PACKAGES = [
    # Core dependencies
    "torch>=2.0.0",
    "torchvision>=0.15.1",
    "numpy>=1.24.4",
    
    # SAM2 dependencies
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    
    # BEVNeXt dependencies
    "networkx>=2.2,<2.3",
    "numba>=0.55.0",
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
    ext_modules = []
    
    # BEVNeXt CUDA extensions
    ext_modules.append(
        make_cuda_ext(
            name='bev_pool_v2_ext',
            module='bevnext.ops.bev_pool_v2',
            sources=[
                'src/bev_pool.cpp',
            ],
            sources_cuda=[
                'src/bev_pool_cuda.cu',
            ],
        )
    )
    
    # SAM2 CUDA extensions
    ext_modules.extend(get_sam2_extensions())
    
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
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    ) 