"""
Baselines package
"""
from .transformer_baselines import (
    UnimodalModel,
    ConcatFusion,
    VanillaCMT,
    ScalarGateFusion
)

__all__ = [
    'UnimodalModel',
    'ConcatFusion',
    'VanillaCMT',
    'ScalarGateFusion'
]
