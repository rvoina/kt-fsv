"""
kt-fsv package.

Feature Selective Validation (FSV) implementation per IEEE Std 1597.1-2022.

Version: 0.2.0
"""

from .fsv import FSV
from .config import FSVConfig
from .analysis import FSVAnalysis, FSVDebugData, get_category
from .visualization import FSVVisualizer

# Optional pre/post-processing modules for advanced use cases
from . import postprocessing, preprocessing
from .preprocessing import build_working_data_sets, WorkingDataSets

__version__ = "0.2.0"
__all__ = [
    "FSV",
    "FSVConfig",
    "FSVAnalysis",
    "FSVDebugData",
    "FSVVisualizer",
    "get_category",
    "postprocessing",
    "preprocessing",
    "build_working_data_sets",
    "WorkingDataSets",
]