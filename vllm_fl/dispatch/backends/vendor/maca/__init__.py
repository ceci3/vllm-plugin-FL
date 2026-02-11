# Copyright (c) 2026 BAAI. All rights reserved.

"""
MACA backend for vllm-plugin-FL dispatch.
"""

from .maca import MacaBackend
from . import patches

__all__ = [
    "MacaBackend",
    "patches",
    ]
