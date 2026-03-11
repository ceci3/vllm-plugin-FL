# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference activation operator implementations using PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_torch(x: torch.Tensor, obj=None) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using PyTorch.

    Args:
        x: Input tensor of shape [..., 2*d]
        obj: The calling obj (optional, for interface consistency)

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2


def gelu_and_mul_torch(x: torch.Tensor, obj=None) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using PyTorch.

    Args:
        x: Input tensor of shape [..., 2*d]
        obj: The calling obj (optional, for interface consistency)

    Returns:
        Output tensor of shape [..., d]
    """
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2
