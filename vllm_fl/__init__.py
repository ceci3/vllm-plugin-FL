# Copyright (c) 2025 BAAI. All rights reserved.


import os


def register():
    """Register the FL platform."""

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry
    from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM
    ModelRegistry.register_model(
        "Qwen3NextForCausalLM",
        "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM")