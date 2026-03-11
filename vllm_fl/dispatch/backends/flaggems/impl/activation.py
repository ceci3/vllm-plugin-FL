# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_flaggems(x: torch.Tensor, obj=None) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        x: Input tensor of shape [..., 2*d]
        obj: The calling obj (optional, for interface consistency)

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(x: torch.Tensor, obj=None) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        x: Input tensor of shape [..., 2*d]
        obj: The calling obj (optional, for interface consistency)

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.fused import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)
