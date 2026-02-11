# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.layernorm import rms_norm, fused_add_rms_norm

import torch

def rms_norm_maca(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation.
    """
    add_residual = residual is not None
    if add_residual:
        return fused_add_rms_norm(
            x, residual, weight, epsilon
        )
    else:
        return rms_norm(x, weight, epsilon)