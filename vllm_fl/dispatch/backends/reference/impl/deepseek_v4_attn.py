# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference (PyTorch) implementations for DeepseekV4 attention operators.
"""

import torch


def deepseek_v4_fp8_einsum_torch(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    """
    Reference implementation of deepseek_v4_fp8_einsum using vLLM's fp8_einsum.

    Falls back to the same deep_gemm utility since there's no pure-PyTorch
    equivalent for FP8 grouped einsum.

    Mutates `out` in-place.
    """
    from vllm.utils.deep_gemm import fp8_einsum

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """
    Reference implementation of fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

    Falls back to the CUDA custom op since there's no pure-PyTorch decomposition
    available for the fused qnorm + rope + kv rope + quant + insert operation.

    Mutates q, swa_kv_cache_2d in-place.
    """
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, swa_kv_cache_2d, slot_mapping, positions, cos_sin_cache,
        eps, block_size,
    )


# ==================== Sparse Attention Indexer Ops ====================


def combine_topk_swa_indices_torch(
    topk_indices: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_tokens: int,
    window_size: int,
    compress_ratio: int,
    block_size: int,
) -> None:
    """Reference implementation of combine_topk_swa_indices via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import combine_topk_swa_indices

    combine_topk_swa_indices(
        topk_indices, combined_indices, combined_lens,
        query_start_loc, seq_lens, block_table,
        topk_tokens, window_size, compress_ratio, block_size,
    )


def compute_global_topk_indices_and_lens_torch(
    topk_indices: torch.Tensor,
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_tokens: int,
    compress_ratio: int,
    block_size: int,
) -> None:
    """Reference implementation of compute_global_topk_indices_and_lens via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import compute_global_topk_indices_and_lens

    compute_global_topk_indices_and_lens(
        topk_indices, global_indices, global_lens,
        query_start_loc, seq_lens, block_table,
        topk_tokens, compress_ratio, block_size,
    )


def dequantize_and_gather_k_cache_torch(
    k_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
) -> None:
    """Reference implementation of dequantize_and_gather_k_cache via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import dequantize_and_gather_k_cache

    dequantize_and_gather_k_cache(
        k_cache, dst, block_table, cu_seq_lens, block_size,
    )


def fused_indexer_q_rope_quant_torch(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    """Reference implementation of fused_indexer_q_rope_quant via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import fused_indexer_q_rope_quant

    return fused_indexer_q_rope_quant(
        positions, index_q, index_q_cos_sin_cache,
        index_weights, index_weights_softmax_scale,
        index_weights_head_scale, use_fp4,
    )


def fused_inv_rope_fp8_quant_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
):
    """Reference implementation of fused_inv_rope_fp8_quant via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import fused_inv_rope_fp8_quant

    return fused_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        heads_per_group, quant_group_size, chunks_per_head,
        rope_start, half_rope, tma_aligned_scales, fp8_max,
        tma_aligned_T, num_tokens, n_groups, d, scale_inner,
    )


def fused_q_kv_rmsnorm_torch(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of fused_q_kv_rmsnorm via upstream Triton kernel."""
    from vllm.v1.attention.ops.deepseek_v4_ops import fused_q_kv_rmsnorm

    return fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def indexer_k_quant_and_cache_torch(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Reference implementation of indexer_k_quant_and_cache via CUDA custom op."""
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
    )


def cp_gather_indexer_k_quant_cache_torch(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Reference implementation of cp_gather_indexer_k_quant_cache via CUDA custom op."""
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens,
    )


def top_k_per_row_prefill_torch(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Reference implementation of top_k_per_row_prefill via CUDA custom op."""
    torch.ops._C.top_k_per_row_prefill(
        logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices,
        num_rows, stride0, stride1, topk_tokens,
    )


def pack_seq_triton_torch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
) -> torch.Tensor:
    """Reference implementation of pack_seq_triton via upstream Triton kernel."""
    from vllm.v1.attention.ops.common import pack_seq_triton

    return pack_seq_triton(x, lengths, pad_value)


def top_k_per_row_decode_torch(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Reference implementation of top_k_per_row_decode via CUDA custom op."""
    torch.ops._C.top_k_per_row_decode(
        logits, next_n, seq_lens, raw_topk_indices,
        num_rows, stride0, stride1, topk_tokens,
    )


def unpack_seq_triton_torch(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of unpack_seq_triton via upstream Triton kernel."""
    from vllm.v1.attention.ops.common import unpack_seq_triton

    return unpack_seq_triton(packed_tensor, lengths)
