# Copyright (c) 2025 BAAI. All rights reserved.

import torch
from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul, SiluAndMulWithClamp
from vllm_fl.dispatch import call_op


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return call_op("silu_and_mul", self, x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return call_op("gelu_and_mul", self, x)


class SiluAndMulWithClampFL(SiluAndMulWithClamp):
    def __init__(self, swiglu_limit: float):
        super().__init__(swiglu_limit)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return call_op("silu_and_mul_with_clamp", x, self.swiglu_limit)


__all__ = ["SiluAndMulFL", "GeluAndMulFL", "SiluAndMulWithClampFL"]
