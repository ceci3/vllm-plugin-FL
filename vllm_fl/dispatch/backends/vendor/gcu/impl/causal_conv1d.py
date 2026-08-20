# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU launch configuration overrides for vLLM causal-conv kernels."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _BlockNKernelProxy:
    """Override ``BLOCK_N`` while preserving the wrapped Triton kernel API."""

    def __init__(self, kernel: Any, block_n: int) -> None:
        self._kernel = kernel
        self.block_n = block_n

    def __getitem__(self, grid: Any) -> Any:
        launch = self._kernel[grid]

        def launch_with_gcu_block_n(*args: Any, **kwargs: Any) -> Any:
            kwargs["BLOCK_N"] = self.block_n
            return launch(*args, **kwargs)

        return launch_with_gcu_block_n

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


def _wrap_kernel(module: Any, name: str, block_n: int) -> None:
    kernel = getattr(module, name)
    if isinstance(kernel, _BlockNKernelProxy):
        if kernel.block_n != block_n:
            raise RuntimeError(
                f"GCU causal-conv kernel {name} already has BLOCK_N="
                f"{kernel.block_n}, expected {block_n}."
            )
        return
    setattr(module, name, _BlockNKernelProxy(kernel, block_n))


def apply_causal_conv1d_gcu_patch() -> None:
    """Apply the S60-tuned causal-conv launch tiles to vLLM."""
    causal_conv1d = importlib.import_module(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    )
    _wrap_kernel(causal_conv1d, "_causal_conv1d_fwd_kernel", block_n=2048)
    _wrap_kernel(causal_conv1d, "_causal_conv1d_update_kernel", block_n=1024)
    logger.info(
        "Patched causal-conv launch tiles for GCU "
        "(prefill BLOCK_N=2048, decode BLOCK_N=1024)"
    )
