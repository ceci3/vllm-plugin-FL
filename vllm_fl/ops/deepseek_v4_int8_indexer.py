# SPDX-License-Identifier: Apache-2.0
"""Direct INT8 kernels for the DeepSeek V4 sparse indexer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round_to_int8(x):
    """Symmetric round-half-away-from-zero, then clamp to the INT8 range.

    ``x.to(tl.int8)`` truncates toward zero, which biases every quantized
    magnitude downward by up to a full LSB instead of half of one. Adding the
    signed 0.5 offset before the cast recovers round-to-nearest; ``tl.clamp``
    runs afterwards so the offset itself cannot push a value out of range.
    Written with plain arithmetic rather than libdevice ``rint`` to stay
    portable across the vendor backends this plugin targets.
    """
    return tl.clamp(
        x + tl.where(x >= 0, 0.5, -0.5), -127.0, 127.0
    ).to(tl.int8)


@triton.jit
def fused_compress_rope_int8_indexer_cache_kernel(
    state_cache,
    state_stride_block,
    state_stride_token,
    token_to_req,
    positions,
    state_slots,
    state_block_table,
    state_block_table_stride,
    state_block_size,
    norm_weight,
    norm_eps,
    cos_sin,
    cos_sin_stride,
    k_cache,
    kv_slots,
    kv_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Compress and directly write INT8 indexer K without an FP8 stage."""
    token = tl.program_id(0)
    state_slot = tl.load(state_slots + token)
    if state_slot < 0:
        return
    position = tl.load(positions + token)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    kv_slot = tl.load(kv_slots + token)
    if kv_slot < 0:
        return

    request = tl.load(token_to_req + token)
    count: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    history = tl.arange(0, count)
    history_pos = position - count + 1 + history
    history_valid = history_pos >= 0
    logical_blocks = history_pos // state_block_size
    physical_blocks = tl.load(
        state_block_table
        + request * state_block_table_stride
        + logical_blocks,
        mask=history_valid,
        other=0,
    ).to(tl.int64)
    block_offsets = history_pos % state_block_size
    overlap_offset = (history >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    dims = tl.arange(0, HEAD_SIZE)
    rows = (
        state_cache
        + physical_blocks * state_stride_block
        + block_offsets * state_stride_token
        + overlap_offset
    )
    valid = history_valid[:, None]
    scores = tl.load(
        rows[:, None] + STATE_WIDTH + dims[None, :],
        mask=valid,
        other=float("-inf"),
    )
    scores = tl.softmax(scores, dim=0)
    values = tl.load(
        rows[:, None] + dims[None, :], mask=valid, other=0.0
    )
    compressed = tl.sum(values * scores, axis=0)

    weight = tl.load(norm_weight + dims)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normalized = compressed * tl.rsqrt(variance + norm_eps) * weight

    half_rope: tl.constexpr = ROPE_HEAD_DIM // 2
    nope_dim: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    normalized_pairs = tl.reshape(normalized, (HEAD_SIZE // 2, 2))
    normalized_even, normalized_odd = tl.split(normalized_pairs)
    partner = tl.interleave(normalized_odd, normalized_even)
    rope_local = dims - nope_dim
    is_rope = dims >= nope_dim
    pair = tl.maximum(rope_local >> 1, 0)
    compressed_position = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cs = cos_sin + compressed_position * cos_sin_stride
    cos_v = tl.load(cs + pair, mask=is_rope, other=1.0)
    sin_v = tl.load(cs + half_rope + pair, mask=is_rope, other=0.0)
    is_even = (rope_local & 1) == 0
    rotated = tl.where(
        is_even,
        normalized * cos_v - partner * sin_v,
        normalized * cos_v + partner * sin_v,
    )
    result = tl.where(is_rope, rotated, normalized).to(tl.bfloat16).to(
        tl.float32
    )

    absmax = tl.maximum(tl.max(tl.abs(result), axis=0), 1.0e-4)
    scale = absmax / 127.0
    quant = _round_to_int8(result / scale)

    kv_block = kv_slot // kv_block_size
    kv_pos = kv_slot % kv_block_size
    block_base = kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    data_base = block_base + kv_pos * HEAD_SIZE
    tl.store(k_cache + data_base + dims, quant)
    scale_byte = block_base + kv_block_size * HEAD_SIZE + kv_pos * 4
    tl.store(
        k_cache.to(tl.pointer_type(tl.float32)) + scale_byte // 4, scale
    )


@triton.jit
def _indexer_q_rope_int8_kernel(
    positions,
    q,
    q_stride_t,
    q_stride_h,
    cos_sin,
    cos_sin_stride,
    q_int8,
    q_int8_stride_t,
    q_int8_stride_h,
    weights,
    weights_stride,
    weights_out,
    weights_out_stride,
    softmax_scale,
    head_scale,
    HEAD_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    nope_dim: tl.constexpr = HEAD_DIM - 2 * HALF_ROPE
    offsets = tl.arange(0, HEAD_DIM)
    base = q + token * q_stride_t + head * q_stride_h
    x = tl.load(base + offsets).to(tl.float32)

    position = tl.load(positions + token)
    cache = cos_sin + position * cos_sin_stride
    rope_local = offsets - nope_dim
    is_rope = offsets >= nope_dim
    partner = tl.load(base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    even = (rope_local & 1) == 0
    rotated = tl.where(
        even,
        x * cos_v - partner * sin_v,
        x * cos_v + partner * sin_v,
    )
    x = tl.where(is_rope, rotated, x).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scale = absmax / 127.0
    quant = _round_to_int8(x / scale)
    out_base = q_int8 + token * q_int8_stride_t + head * q_int8_stride_h
    tl.store(out_base + offsets, quant)

    weight = tl.load(weights + token * weights_stride + head).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_stride + head,
        weight * scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant_int8(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply GPT-J RoPE and directly quantize indexer Q to INT8.

    The per-token/head Q scale is folded into ``weights_out``, matching the
    existing FP8 indexer contract while avoiding any FP8 representation.
    """
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    _indexer_q_rope_int8_kernel[(positions.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_int8,
        q_int8.stride(0),
        q_int8.stride(1),
        weights,
        weights.stride(0),
        weights_out,
        weights_out.stride(0),
        softmax_scale,
        head_scale,
        HEAD_DIM=q.shape[2],
        HALF_ROPE=cos_sin_cache.shape[1] // 2,
        num_warps=1,
    )
    return q_int8, weights_out


@triton.jit
def _int8_mqa_logits_h64_d128_kernel(
    q,
    k,
    k_scale,
    weights,
    cu_ks,
    cu_ke,
    logits,
    num_rows,
    num_keys,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """DSV4 H64/D128 INT8 MQA with a fused query/head Tensor Core tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    keys = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    key_mask = keys < num_keys
    dims = tl.arange(0, 128)
    heads = tl.arange(0, 64)

    # Flatten [BLOCK_M, H, D] into the GEMM N dimension. This follows
    # DeepGEMM's BLOCK_Q * num_heads WGMMA mapping and reuses each K tile for
    # multiple queries and all 64 heads.
    q_rows = tl.arange(0, BLOCK_M * 64)
    query_rows = pid_m * BLOCK_M + q_rows // 64
    query_heads = q_rows % 64
    q_tile = tl.load(
        q
        + query_rows[:, None] * (64 * 128)
        + query_heads[:, None] * 128
        + dims[None, :],
        mask=(query_rows < num_rows)[:, None],
        other=0,
        eviction_policy="evict_last",
    )
    k_tile = tl.load(
        k + keys[:, None] * 128 + dims[None, :],
        mask=key_mask[:, None],
        other=0,
        eviction_policy="evict_first",
    )
    scales = tl.load(k_scale + keys, mask=key_mask, other=0.0)

    dots = tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.int32)
    scores = tl.reshape(dots, (BLOCK_N, BLOCK_M, 64)).to(tl.float32)
    head_weights = tl.load(
        weights + rows[:, None] * 64 + heads[None, :],
        mask=row_mask[:, None],
        other=0.0,
        eviction_policy="evict_last",
    )
    values = tl.sum(
        tl.maximum(scores * scales[:, None, None], 0.0)
        * head_weights[None, :, :],
        axis=2,
    )

    starts = tl.load(cu_ks + rows, mask=row_mask, other=0)
    ends = tl.load(cu_ke + rows, mask=row_mask, other=0)
    valid = (
        row_mask[:, None]
        & key_mask[None, :]
        & (keys[None, :] >= starts[:, None])
        & (keys[None, :] < ends[:, None])
    )
    tl.store(
        logits + rows[:, None] * num_keys + keys[None, :],
        tl.trans(values),
        mask=valid,
        eviction_policy="evict_first",
    )


@triton.jit
def _int8_mqa_logits_tensor_core_kernel(
    q,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    k,
    k_stride_n,
    k_stride_d,
    k_scale,
    weights,
    weights_stride_m,
    weights_stride_h,
    cu_ks,
    cu_ke,
    logits,
    logits_stride_m,
    num_rows,
    num_keys,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Stride-aware parameterized INT8 Tensor Core MQA fallback."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    keys = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    key_mask = keys < num_keys
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM

    # Pad heads/dims to Tensor Core-compatible powers of two. Padded query
    # rows and weights are zero, so they disappear in the head reduction.
    q_rows = tl.arange(0, BLOCK_M * BLOCK_H)
    query_rows = pid_m * BLOCK_M + q_rows // BLOCK_H
    query_heads = q_rows % BLOCK_H
    q_tile = tl.load(
        q
        + query_rows[:, None] * q_stride_m
        + query_heads[:, None] * q_stride_h
        + dims[None, :] * q_stride_d,
        mask=(query_rows < num_rows)[:, None]
        & (query_heads < NUM_HEADS)[:, None]
        & dim_mask[None, :],
        other=0,
        eviction_policy="evict_last",
    )
    k_tile = tl.load(
        k + keys[:, None] * k_stride_n + dims[None, :] * k_stride_d,
        mask=key_mask[:, None] & dim_mask[None, :],
        other=0,
        eviction_policy="evict_first",
    )
    scales = tl.load(k_scale + keys, mask=key_mask, other=0.0)

    dots = tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.int32)
    scores = tl.reshape(dots, (BLOCK_N, BLOCK_M, BLOCK_H)).to(
        tl.float32
    )
    heads = tl.arange(0, BLOCK_H)
    head_weights = tl.load(
        weights
        + rows[:, None] * weights_stride_m
        + heads[None, :] * weights_stride_h,
        mask=row_mask[:, None] & (heads < NUM_HEADS)[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    values = tl.sum(
        tl.maximum(scores * scales[:, None, None], 0.0)
        * head_weights[None, :, :],
        axis=2,
    )

    starts = tl.load(cu_ks + rows, mask=row_mask, other=0)
    ends = tl.load(cu_ke + rows, mask=row_mask, other=0)
    valid = (
        row_mask[:, None]
        & key_mask[None, :]
        & (keys[None, :] >= starts[:, None])
        & (keys[None, :] < ends[:, None])
    )
    tl.store(
        logits
        + rows[:, None] * logits_stride_m
        + keys[None, :],
        tl.trans(values),
        mask=valid,
        eviction_policy="evict_first",
    )


def int8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_ks: torch.Tensor,
    cu_ke: torch.Tensor,
) -> torch.Tensor:
    """Contiguous INT8 indexer logits used by prefill."""
    num_keys = k.shape[0]
    logits = torch.empty(
        (q.shape[0], num_keys), dtype=torch.float32, device=q.device
    )
    if (
        q.shape[1] == 64
        and q.shape[2] == 128
        and q.is_contiguous()
        and k.is_contiguous()
        and weights.is_contiguous()
    ):
        block_m = 2
        block_n = 64
        _int8_mqa_logits_h64_d128_kernel[
            (triton.cdiv(q.shape[0], block_m), triton.cdiv(num_keys, block_n))
        ](
            q,
            k,
            k_scale,
            weights,
            cu_ks,
            cu_ke,
            logits,
            q.shape[0],
            num_keys,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
            num_stages=1,
        )
        return logits

    block_h = triton.next_power_of_2(q.shape[1])
    block_d = max(32, triton.next_power_of_2(q.shape[2]))
    # Match DeepGEMM's 128 query-head rows per MMA tile where possible.
    block_m = max(1, 128 // block_h)
    block_n = 64
    _int8_mqa_logits_tensor_core_kernel[
        (triton.cdiv(q.shape[0], block_m), triton.cdiv(num_keys, block_n))
    ](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k,
        k.stride(0),
        k.stride(1),
        k_scale,
        weights,
        weights.stride(0),
        weights.stride(1),
        cu_ks,
        cu_ke,
        logits,
        logits.stride(0),
        q.shape[0],
        num_keys,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=1,
    )
    return logits


@triton.jit
def _int8_paged_mqa_logits_h64_d128_kernel(
    q,
    cache,
    cache_block_stride,
    weights,
    context_lens,
    context_row_stride,
    context_col_stride,
    block_table,
    block_table_stride,
    logits,
    max_model_len,
    NEXT_N: tl.constexpr,
):
    """DSV4-specialized H=64, D=128, page=64 decode kernel.

    This mirrors FlagGems' BF16 specialization: source-level shape literals,
    one physical page per program, one page-table lookup, unmasked full-page
    loads, and cache hints matching the reuse distance of Q/K.
    """
    row = tl.program_id(0)
    logical_block = tl.program_id(1)
    batch = row // NEXT_N
    next_idx = row % NEXT_N
    # Strides come from the caller: a compact (B, 1) ``context_lens`` passes a
    # zero column stride so the shared length broadcasts across next_n.
    context_len = tl.load(
        context_lens
        + batch * context_row_stride
        + next_idx * context_col_stride
    )
    key_start = logical_block * 64
    if key_start >= context_len:
        return

    heads = tl.arange(0, 64)
    dims = tl.arange(0, 128)
    positions = tl.arange(0, 64)

    # Decode Q and weights are contiguous and reused by every cache page.
    q_tile = tl.load(
        q + row * (64 * 128) + heads[:, None] * 128 + dims[None, :],
        eviction_policy="evict_last",
    )
    head_weights = tl.load(
        weights + row * 64 + heads, eviction_policy="evict_last"
    )

    physical_block = tl.load(
        block_table + batch * block_table_stride + logical_block
    ).to(tl.int64)
    page_base = physical_block * cache_block_stride
    k_tile = tl.load(
        cache + page_base + positions[:, None] * 128 + dims[None, :],
        eviction_policy="evict_first",
    ).to(tl.int8)
    k_scales = tl.load(
        cache.to(tl.pointer_type(tl.float32))
        + (page_base + 64 * 128) // 4
        + positions,
        eviction_policy="evict_first",
    )

    dots = tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.int32)
    activated = tl.maximum(dots.to(tl.float32) * k_scales[:, None], 0.0)
    result = tl.sum(activated * head_weights[None, :], axis=1)
    output = logits + row * max_model_len + key_start + positions
    if key_start + 64 <= context_len:
        tl.store(output, result, eviction_policy="evict_first")
    else:
        tl.store(
            output,
            result,
            mask=positions < context_len - key_start,
            eviction_policy="evict_first",
        )


@triton.jit
def _int8_paged_mqa_logits_kernel(
    q,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    cache,
    cache_block_stride,
    block_size,
    weights,
    weights_stride,
    context_lens,
    context_row_stride,
    context_col_stride,
    block_table,
    block_table_stride,
    logits,
    logits_stride,
    max_model_len,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    batch = row // NEXT_N
    next_idx = row % NEXT_N
    # See int8_paged_mqa_logits: a compact (B, 1) ``context_lens`` passes a zero
    # column stride so the shared length broadcasts across next_n positions.
    context_len = tl.load(
        context_lens
        + batch * context_row_stride
        + next_idx * context_col_stride
    )

    # The logits buffer is deliberately not cleaned: the downstream top-k
    # kernel is length-masked.  Decode allocates it at max_model_len, which can
    # be tens of thousands for DSV4, while a request commonly has only a few
    # cache blocks.  Returning before any page-table/cache access is critical;
    # otherwise every padded tile performs a complete 64-head dot product.
    key_start = key_block * BLOCK_KEYS
    if key_start >= context_len:
        return
    keys = key_start + tl.arange(0, BLOCK_KEYS)
    key_mask = (keys < context_len) & (keys < max_model_len)
    logical_blocks = keys // block_size
    block_offsets = keys % block_size
    physical_blocks = tl.load(
        block_table + batch * block_table_stride + logical_blocks,
        mask=key_mask,
        other=0,
    ).to(tl.int64)
    byte_bases = physical_blocks * cache_block_stride + block_offsets * HEAD_DIM
    dims = tl.arange(0, HEAD_DIM)
    k_tile = tl.load(
        cache + byte_bases[:, None] + dims[None, :],
        mask=key_mask[:, None],
        other=0,
    ).to(tl.int8)
    scale_offsets = (
        physical_blocks * cache_block_stride
        + block_size * HEAD_DIM
        + block_offsets * 4
    )
    k_scales = tl.load(
        cache.to(tl.pointer_type(tl.float32)) + scale_offsets // 4,
        mask=key_mask,
        other=0.0,
    )

    acc = tl.zeros((BLOCK_KEYS,), tl.float32)
    # Map INT8 K@Q to Hopper tensor cores.  The previous scalar-head loop cast
    # both operands to FP32 and emitted 64 independent reductions per key
    # tile. Tiling heads makes the contraction a native INT8 dot producing
    # exact INT32 accumulators; scales/ReLU/weights remain FP32 as required by
    # the indexer definition.
    for head_start in tl.static_range(0, NUM_HEADS, BLOCK_HEADS):
        heads = head_start + tl.arange(0, BLOCK_HEADS)
        head_mask = heads < NUM_HEADS
        q_tile = tl.load(
            q
            + batch * q_stride_b
            + next_idx * q_stride_n
            + heads[:, None] * q_stride_h
            + dims[None, :],
            mask=head_mask[:, None],
            other=0,
        )
        dots = tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.int32)
        head_weights = tl.load(
            weights + row * weights_stride + heads,
            mask=head_mask,
            other=0.0,
        )
        activated = tl.maximum(
            dots.to(tl.float32) * k_scales[:, None], 0.0
        )
        acc += tl.sum(activated * head_weights[None, :], axis=1)
    tl.store(logits + row * logits_stride + keys, acc, mask=key_mask)


def int8_paged_mqa_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Paged INT8 indexer logits used by decode."""
    batch, next_n, num_heads, head_dim = q.shape
    logits = torch.empty(
        (batch * next_n, max_model_len), dtype=torch.float32, device=q.device
    )
    flat = cache.view(torch.uint8)

    # ``context_lens`` is (B, next_n) under native spec decode but (B, 1)
    # otherwise. The kernels index it as ``batch * row_stride + next_idx``, so a
    # compact single-column tensor must advertise a zero column stride to
    # broadcast the shared length across the next_n speculative positions --
    # taking stride(0) alone would walk into the following request's entry.
    context_row_stride = context_lens.stride(0)
    if context_lens.dim() > 1 and context_lens.shape[1] == 1:
        context_col_stride = 0
    elif context_lens.dim() == 1:
        context_row_stride, context_col_stride = context_lens.stride(0), 0
    else:
        context_col_stride = context_lens.stride(1)
    if (
        num_heads == 64
        and head_dim == 128
        and cache.shape[1] == 64
        and q.is_contiguous()
        and weights.is_contiguous()
        and context_lens.is_contiguous()
    ):
        # A context cannot address more pages than its block-table row. Avoid
        # launching capacity-only CTAs when metadata carries a compact table.
        grid_blocks = min(
            triton.cdiv(max_model_len, 64), block_table.shape[1]
        )
        _int8_paged_mqa_logits_h64_d128_kernel[
            (batch * next_n, grid_blocks)
        ](
            q,
            flat,
            cache.stride(0),
            weights,
            context_lens,
            context_row_stride,
            context_col_stride,
            block_table,
            block_table.stride(0),
            logits,
            max_model_len,
            NEXT_N=next_n,
            num_warps=8,
            num_stages=1,
        )
        return logits

    _int8_paged_mqa_logits_kernel[
        (batch * next_n, triton.cdiv(max_model_len, 128))
    ](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        flat,
        cache.stride(0),
        cache.shape[1],
        weights,
        weights.stride(0),
        context_lens,
        context_row_stride,
        context_col_stride,
        block_table,
        block_table.stride(0),
        logits,
        logits.stride(0),
        max_model_len,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KEYS=128,
        BLOCK_HEADS=32,
        num_warps=8,
    )
    return logits
