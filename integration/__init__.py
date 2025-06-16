"""
BEVNeXt-SAM2 Integration Module

This module provides integration between BEVNeXt and SAM2 for enhanced 3D object detection
with segmentation capabilities.
"""

from .bev_sam_fusion import BEVSAMFusion
from .sam_enhanced_detector import SAMEnhancedBEVDetector

__all__ = ['BEVSAMFusion', 'SAMEnhancedBEVDetector'] 