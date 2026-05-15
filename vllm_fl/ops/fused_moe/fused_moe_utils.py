
from enum import Enum
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend, map_unquantized_backend
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.utils import _resize_cache, moe_kernel_quantize_input
from vllm.triton_utils import tl, triton
from vllm_fl.dispatch import call_op
from vllm_fl.ops.fused_moe.activation import apply_moe_activation

logger = init_logger(__name__)

## Adopt from select_unquantized_moe_backend
def select_unquantized_moe_backend_oot(moe_config: FusedMoEConfig, use_ep: bool, use_dp: bool):
    """
    Select the primary Unquantized MoE backend and experts class.
    Returns: (backend, experts_cls)
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
    flashinfer_cutlass_available = (
        has_flashinfer_cutlass_fused_moe()
        and use_ep
        and (not use_dp)
        and current_platform.has_device_capability(90)
    )
    flashinfer_cutlass_moe_enabled = (
        flashinfer_cutlass_available and envs.VLLM_USE_FLASHINFER_MOE_FP16
    )
    rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    # Handle explicit moe_backend from user.
    runner_backend = moe_config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_unquantized_backend(runner_backend)
        if requested_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
            if not flashinfer_cutlass_available:
                raise ValueError(
                    "FlashInfer CUTLASS MoE backend is not available for this "
                    "configuration."
                )
        elif requested_backend == UnquantizedMoeBackend.AITER and not (
            current_platform.is_rocm() and rocm_aiter_moe_enabled
        ):
            raise ValueError(
                "ROCm AITer MoE backend is not available for this configuration."
            )
        logger.info_once(_make_log_backend(requested_backend))
        return requested_backend, TritonExpertsFL

    if current_platform.is_rocm():
        if rocm_aiter_moe_enabled:
            backend = UnquantizedMoeBackend.AITER
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_cuda() or current_platform.is_cuda_alike():
        if flashinfer_cutlass_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_xpu():
        backend = UnquantizedMoeBackend.XPU
    if current_platform.is_cpu():
        backend = UnquantizedMoeBackend.CPU
    if current_platform.is_tpu():
        backend = UnquantizedMoeBackend.TPU

    logger.info_once(_make_log_backend(backend))
    return backend, TritonExpertsFL

def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict[str, Any],
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    *,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    block_shape: list[int] | None = None,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Prepare expert assignments for the aligned and low-latency Triton paths."""
    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
    # activates only a small fraction of total experts
    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
    # path of the fused_moe_kernel kernel
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * 4 <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return call_op("moe_align_block_size",
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )

class TritonExpertsFL(TritonExperts):
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )

        call_op("invoke_fused_moe_triton_kernel",
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        # self.activation(
        apply_moe_activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

        call_op("invoke_fused_moe_triton_kernel",
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        call_op("moe_sum", input, output)