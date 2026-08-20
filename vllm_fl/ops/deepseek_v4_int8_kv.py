# SPDX-License-Identifier: Apache-2.0
"""Triton helpers for DeepSeek V4 per-token-head INT8 KV caches."""

import torch

from vllm.triton_utils import tl, triton

# DeepSeek-V4 INT8 page layout, shared by the writers and readers below.
# Declared as ``tl.constexpr`` so the @triton.jit kernels can reference them
# directly (plain module-level ints are rejected by Triton); use ``.value`` for
# host-side arithmetic.
#
# Per-token page (608B):
#   [0, 448)      448 x int8   NoPE
#   [448, 576)    64 x bfloat16 RoPE
#   [576, 608)    7 x float32 scales + 4B pad (64 NoPE channels per scale)
_NOPE_DIM = tl.constexpr(448)
_HEAD_DIM = tl.constexpr(512)
_TOKEN_DATA_BYTES = tl.constexpr(576)
_QUANT_BLOCK_SIZE = tl.constexpr(64)
_SCALE_SLOT_BYTES = tl.constexpr(32)
# Offset of the RoPE half when the page is viewed as bfloat16 (448 / 2).
_ROPE_BF16_OFFSET = tl.constexpr(224)

# Plain-int mirror for host-side callers (KV cache shape/spec construction),
# which cannot index with a tl.constexpr.
INT8_TOKEN_PAGE_BYTES = 608


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
def fused_compress_rope_int8_mla_cache_kernel(
    state_cache, state_stride_block, state_stride_token,
    token_to_req, positions, state_slots, state_block_table,
    state_block_table_stride, state_block_size, norm_weight, norm_eps,
    cos_sin, cos_sin_stride, k_cache, kv_slots, kv_block_size,
    HEAD_SIZE: tl.constexpr, TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr, QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr, SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr, STORE_FP8: tl.constexpr = False,
):
    """Compress directly to 448 INT8 NoPE + 64 BF16 RoPE."""
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
    physical_blocks = tl.load(
        state_block_table + request * state_block_table_stride
        + history_pos // state_block_size,
        mask=history_valid, other=0,
    ).to(tl.int64)
    block_offsets = history_pos % state_block_size
    overlap_offset = (history >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    dims = tl.arange(0, TRITON_BLOCK_SIZE)
    dim_mask = dims < HEAD_SIZE
    rows = (state_cache + physical_blocks * state_stride_block
            + block_offsets * state_stride_token + overlap_offset)
    valid = history_valid[:, None] & dim_mask[None, :]
    scores = tl.load(rows[:, None] + STATE_WIDTH + dims[None, :],
                     mask=valid, other=float("-inf"))
    scores = tl.softmax(scores, dim=0)
    values = tl.load(rows[:, None] + dims[None, :], mask=valid, other=0.0)
    compressed = tl.sum(values * scores, axis=0)
    weight = tl.load(norm_weight + dims, mask=dim_mask, other=0.0)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normalized = compressed * tl.rsqrt(variance + norm_eps) * weight
    normalized = normalized.to(tl.bfloat16).to(tl.float32)

    nope_dim: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    nope_mask = dims < nope_dim
    normalized_blocks = tl.reshape(
        normalized, (TRITON_BLOCK_SIZE // QUANT_BLOCK, QUANT_BLOCK)
    )
    absmax = tl.maximum(tl.max(tl.abs(normalized_blocks), axis=1), 1.0e-4)
    if STORE_FP8:
        raw_scale = absmax / 448.0
        scale_bits = raw_scale.to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + (
            (scale_bits & 0x7FFFFF) != 0
        ).to(tl.uint32)
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        scales = (scale_exp << 23).to(tl.float32, bitcast=True)
    else:
        scales = tl.maximum(absmax / 127.0, 1.0e-12)
    scale_for_dim = tl.reshape(
        tl.broadcast_to(scales[:, None], normalized_blocks.shape),
        (TRITON_BLOCK_SIZE,),
    )
    if STORE_FP8:
        quant = (normalized / scale_for_dim).to(tl.float8e4nv)
    else:
        quant = _round_to_int8(normalized / scale_for_dim)
    kv_block = kv_slot // kv_block_size
    kv_pos = kv_slot % kv_block_size
    block_base = kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    data_base = block_base + kv_pos * TOKEN_STRIDE
    if STORE_FP8:
        tl.store(
            (k_cache + data_base + dims).to(
                tl.pointer_type(tl.float8e4nv)
            ), quant, mask=nope_mask,
        )
    else:
        tl.store(k_cache + data_base + dims, quant, mask=nope_mask)

    pairs = tl.reshape(normalized, (TRITON_BLOCK_SIZE // 2, 2))
    even, odd = tl.split(pairs)
    pair_idx = tl.arange(0, TRITON_BLOCK_SIZE // 2)
    rope_pair = pair_idx - nope_dim // 2
    is_rope_pair = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs = cos_sin + (position // COMPRESS_RATIO) * COMPRESS_RATIO * cos_sin_stride
    cos_v = tl.load(cs + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cs + ROPE_HEAD_DIM // 2 + cs_idx,
                    mask=is_rope_pair, other=0.0)
    rotated = tl.interleave(even * cos_v - odd * sin_v,
                            odd * cos_v + even * sin_v)
    rope_local = dims - nope_dim
    rope_ptr = (k_cache + data_base + nope_dim).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + rope_local, rotated.to(tl.bfloat16),
             mask=(dims >= nope_dim) & dim_mask)
    scale_base = (
        block_base
        + kv_block_size * TOKEN_STRIDE
        + kv_pos * _SCALE_SLOT_BYTES
    )
    scale_ids = tl.arange(0, TRITON_BLOCK_SIZE // QUANT_BLOCK)
    tl.store(k_cache.to(tl.pointer_type(tl.float32)) + scale_base // 4
             + scale_ids, scales, mask=scale_ids < nope_dim // QUANT_BLOCK)


@triton.jit
def _qnorm_rope_kv_insert_int8_mla_kernel(
    q, q_stride_t, q_stride_h, kv, kv_stride_t, cache, slots, positions,
    cos_sin, cos_sin_stride, eps, block_size, cache_block_stride,
    NUM_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr, STORE_FP8: tl.constexpr,
):
    """Fuse SWA query preparation and direct INT8 KV insertion.

    One multi-warp program owns a token and walks all query heads before
    writing the shared KV head.  This removes the former 17-program/token
    launch geometry and reuses position/RoPE data across all heads.
    """
    token = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, HEAD_SIZE)
    nope_dim: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    rope_dims = tl.arange(0, ROPE_HEAD_DIM)
    half_rope_dim: tl.constexpr = ROPE_HEAD_DIM // 2
    half_rope_dims = tl.arange(0, half_rope_dim)
    position = tl.load(positions + token)
    cs = cos_sin + position * cos_sin_stride
    cos_v = tl.load(cs + half_rope_dims)
    sin_v = tl.load(cs + half_rope_dim + half_rope_dims)

    for head in range(0, NUM_HEADS):
        ptr = q + token * q_stride_t + head * q_stride_h
        original = tl.load(ptr + dims).to(tl.float32)
        rrms = tl.rsqrt(
            tl.sum(original * original, axis=0) / HEAD_SIZE + eps
        )
        normalized = original * rrms
        tl.store(ptr + dims, normalized.to(tl.bfloat16),
                 mask=dims < nope_dim)
        q_rope = tl.load(ptr + nope_dim + rope_dims).to(tl.float32) * rrms
        q_rope_pairs = tl.reshape(q_rope, (half_rope_dim, 2))
        q_even, q_odd = tl.split(q_rope_pairs)
        q_rotated = tl.reshape(
            tl.join(q_even * cos_v - q_odd * sin_v,
                    q_odd * cos_v + q_even * sin_v),
            (ROPE_HEAD_DIM,),
        ).to(tl.bfloat16)
        tl.store(ptr + nope_dim + rope_dims, q_rotated)

    ptr = kv + token * kv_stride_t
    kv_rope = tl.load(ptr + nope_dim + rope_dims).to(tl.float32)
    kv_rope_pairs = tl.reshape(kv_rope, (half_rope_dim, 2))
    kv_even, kv_odd = tl.split(kv_rope_pairs)
    kv_rotated = tl.reshape(
        tl.join(kv_even * cos_v - kv_odd * sin_v,
                kv_odd * cos_v + kv_even * sin_v),
        (ROPE_HEAD_DIM,),
    ).to(tl.bfloat16)
    slot = tl.load(slots + token).to(tl.int64)
    if slot >= 0:
        x = tl.load(ptr + dims).to(tl.float32)
        nope_mask = dims < nope_dim
        x_blocks = tl.reshape(
            x, (HEAD_SIZE // _QUANT_BLOCK_SIZE, _QUANT_BLOCK_SIZE)
        )
        absmax = tl.maximum(tl.max(tl.abs(x_blocks), axis=1), 1.0e-4)
        if STORE_FP8:
            raw_scale = absmax / 448.0
            bits = raw_scale.to(tl.uint32, bitcast=True)
            exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            scales = (exp << 23).to(tl.float32, bitcast=True)
        else:
            scales = tl.maximum(absmax / 127.0, 1.0e-12)
        scale_for_dim = tl.reshape(
            tl.broadcast_to(scales[:, None], x_blocks.shape), (HEAD_SIZE,)
        )
        if STORE_FP8:
            quant = (x / scale_for_dim).to(tl.float8e4nv)
        else:
            quant = _round_to_int8(x / scale_for_dim)
        block = slot // block_size
        pos = slot % block_size
        block_base = block * cache_block_stride
        data_base = block_base + pos * _TOKEN_DATA_BYTES
        if STORE_FP8:
            tl.store((cache + data_base + dims).to(
                tl.pointer_type(tl.float8e4nv)), quant, mask=nope_mask)
        else:
            tl.store(cache + data_base + dims, quant, mask=nope_mask)
        rope_ptr = (cache + data_base + nope_dim).to(
            tl.pointer_type(tl.bfloat16))
        tl.store(rope_ptr + rope_dims, kv_rotated)
        scale_base = (
            block_base
            + block_size * _TOKEN_DATA_BYTES
            + pos * _SCALE_SLOT_BYTES
        )
        scale_ids = tl.arange(0, HEAD_SIZE // _QUANT_BLOCK_SIZE)
        tl.store(cache.to(tl.pointer_type(tl.float32)) + scale_base // 4
                 + scale_ids, scales,
                 mask=scale_ids < nope_dim // _QUANT_BLOCK_SIZE)


def qnorm_rope_kv_insert_int8_mla(
    q: torch.Tensor, kv: torch.Tensor, cache: torch.Tensor,
    slots: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
    eps: float, block_size: int, store_fp8: bool = False,
) -> None:
    """Apply Q RMSNorm/RoPE and write SWA KV directly as INT8."""
    pos = positions.to(torch.int64)
    flat = cache.view(torch.uint8)
    grid = (q.shape[0],)
    _qnorm_rope_kv_insert_int8_mla_kernel[grid](
        q, q.stride(0), q.stride(1), kv, kv.stride(0), flat, slots, pos,
        cos_sin, cos_sin.stride(0), eps, block_size, cache.stride(0),
        NUM_HEADS=q.shape[1], HEAD_SIZE=q.shape[2], ROPE_HEAD_DIM=64,
        STORE_FP8=store_fp8,
        num_warps=4,
        num_stages=3,
    )


@triton.jit
def _gather_int8_indices_kernel(
    cache_i8,
    cache_bf16,
    cache_f32,
    indices,
    lengths,
    indices_stride,
    out,
    out_indices,
    item_offsets,
    item_offset,
    num_items,
    out_width,
    cache_block_size,
    cache_block_stride,
    has_item_offsets: tl.constexpr,
    BLOCK: tl.constexpr, CACHE_FP8: tl.constexpr,
    OFFICIAL_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    worker = tl.program_id(1)
    num_workers = tl.num_programs(1)
    dims = tl.arange(0, BLOCK)
    is_nope = dims < _NOPE_DIM
    length = tl.load(lengths + row)
    row_item_offset = (
        tl.load(item_offsets + row) if has_item_offsets else item_offset
    )
    # A small persistent worker set replaces one CTA per sparse cache index.
    for item in range(worker, num_items, num_workers):
        valid = item < length
        slot = tl.load(
            indices + row * indices_stride + item, mask=valid, other=0
        ).to(tl.int64)
        dim_mask = valid & (dims < _HEAD_DIM)
        cache_block = slot // cache_block_size
        cache_pos = slot % cache_block_size
        block_base = cache_block * cache_block_stride
        data_base = block_base + cache_pos * _TOKEN_DATA_BYTES
        scale_base = (
            block_base + cache_block_size * _TOKEN_DATA_BYTES
            + cache_pos * _SCALE_SLOT_BYTES
        )
        if OFFICIAL_FP8:
            official_scale_base = (
                block_base + cache_block_size * _TOKEN_DATA_BYTES
                + cache_pos * 8
            )
            scale_byte = tl.load(
                cache_i8.to(tl.pointer_type(tl.uint8))
                + official_scale_base + dims // _QUANT_BLOCK_SIZE,
                mask=valid & is_nope, other=0,
            )
            scale = (scale_byte.to(tl.uint32) << 23).to(
                tl.float32, bitcast=True
            )
        else:
            scale = tl.load(cache_f32 + scale_base // 4
                            + dims // _QUANT_BLOCK_SIZE,
                            mask=valid & is_nope, other=0.0)
        if CACHE_FP8 or OFFICIAL_FP8:
            nope = tl.load((cache_i8 + data_base + dims).to(
                tl.pointer_type(tl.float8e4nv)), mask=dim_mask & is_nope)
        else:
            nope = tl.load(cache_i8 + data_base + dims,
                           mask=dim_mask & is_nope)
        rope = tl.load(
            cache_bf16 + data_base // 2 + _ROPE_BF16_OFFSET
            + (dims - _NOPE_DIM), mask=dim_mask & ~is_nope,
        )
        value = tl.where(is_nope, nope.to(tl.float32) * scale, rope)
        output_item = row_item_offset + item
        out_base = (row * out_width + output_item) * _HEAD_DIM
        tl.store(out + out_base + dims, value, mask=dim_mask)
        tl.store(out_indices + row * out_width + output_item,
                 row * out_width + output_item, mask=valid)
        tl.store(out_indices + row * out_width + output_item, -1, mask=~valid)


def gather_int8_cache_indices(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    out_indices: torch.Tensor,
    item_offset: int = 0,
    item_offsets: torch.Tensor | None = None,
    fp8_cache: bool = False,
    official_fp8_cache: bool = False,
) -> None:
    """Gather sparse slots and dequantize them into a dense workspace."""
    rows, width = indices.shape[0], indices.shape[-1]
    flat = cache.view(torch.uint8)
    # Decode is bandwidth/latency bound and normally has only four rows.
    # 32 workers leave half of H20's 78 SMs idle; 64 workers reduce the
    # production-width gather latency substantially without oversubscribing.
    workers = min(width, 768)
    _gather_int8_indices_kernel[(rows, workers)](
        flat.view(torch.int8),
        flat.view(torch.bfloat16),
        flat.view(torch.float32),
        indices,
        lengths,
        indices.stride(0),
        out,
        out_indices,
        item_offsets,
        item_offset,
        width,
        out.shape[1],
        cache.shape[1],
        cache.stride(0),
        has_item_offsets=item_offsets is not None,
        BLOCK=512, CACHE_FP8=fp8_cache,
        OFFICIAL_FP8=official_fp8_cache,
        num_warps=4,
    )


@triton.jit
def _gather_two_int8_indices_kernel(
    cache0_i8, cache0_bf16, cache0_f32,
    indices0, lengths0, indices0_stride,
    cache1_i8, cache1_bf16, cache1_f32,
    indices1, lengths1, indices1_stride,
    out, out_indices, width0, width1, out_width,
    cache0_block_size, cache0_block_stride,
    cache1_block_size, cache1_block_stride,
    BLOCK: tl.constexpr, CACHE0_FP8: tl.constexpr,
    CACHE1_FP8: tl.constexpr, CACHE1_OFFICIAL_FP8: tl.constexpr,
):
    """Gather main and SWA INT8 caches in one launch during decode."""
    row = tl.program_id(0)
    worker = tl.program_id(1)
    num_workers = tl.num_programs(1)
    dims = tl.arange(0, BLOCK)
    is_nope = dims < _NOPE_DIM
    length0 = tl.load(lengths0 + row)
    length1 = tl.load(lengths1 + row)

    for item in range(worker, width0, num_workers):
        valid = item < length0
        slot = tl.load(
            indices0 + row * indices0_stride + item,
            mask=valid, other=0,
        ).to(tl.int64)
        dim_mask = valid & (dims < _HEAD_DIM)
        cache_block = slot // cache0_block_size
        cache_pos = slot % cache0_block_size
        block_base = cache_block * cache0_block_stride
        data_base = block_base + cache_pos * _TOKEN_DATA_BYTES
        scale_base = (block_base + cache0_block_size * _TOKEN_DATA_BYTES
                      + cache_pos * _SCALE_SLOT_BYTES)
        scale = tl.load(cache0_f32 + scale_base // 4
                        + dims // _QUANT_BLOCK_SIZE,
                        mask=valid & is_nope, other=0.0)
        if CACHE0_FP8:
            nope = tl.load(
                (cache0_i8 + data_base + dims).to(
                    tl.pointer_type(tl.float8e4nv)
                ), mask=dim_mask & is_nope,
            )
        else:
            nope = tl.load(cache0_i8 + data_base + dims,
                           mask=dim_mask & is_nope)
        rope = tl.load(cache0_bf16 + data_base // 2 + _ROPE_BF16_OFFSET
                       + dims - _NOPE_DIM,
                       mask=dim_mask & ~is_nope)
        value = tl.where(is_nope, nope.to(tl.float32) * scale, rope)
        out_base = (row * out_width + item) * _HEAD_DIM
        tl.store(out + out_base + dims, value, mask=dim_mask)
        tl.store(out_indices + row * out_width + item,
                 row * out_width + item, mask=valid)
        tl.store(out_indices + row * out_width + item, -1, mask=~valid)

    for item in range(worker, width1, num_workers):
        valid = item < length1
        slot = tl.load(
            indices1 + row * indices1_stride + item,
            mask=valid, other=0,
        ).to(tl.int64)
        dim_mask = valid & (dims < _HEAD_DIM)
        cache_block = slot // cache1_block_size
        cache_pos = slot % cache1_block_size
        block_base = cache_block * cache1_block_stride
        data_base = block_base + cache_pos * _TOKEN_DATA_BYTES
        scale_base = (block_base + cache1_block_size * _TOKEN_DATA_BYTES
                      + cache_pos * _SCALE_SLOT_BYTES)
        if CACHE1_OFFICIAL_FP8:
            official_scale_base = (
                block_base + cache1_block_size * _TOKEN_DATA_BYTES
                + cache_pos * 8
            )
            scale_byte = tl.load(
                cache1_i8.to(tl.pointer_type(tl.uint8))
                + official_scale_base + dims // _QUANT_BLOCK_SIZE,
                mask=valid & is_nope, other=0,
            )
            scale = (scale_byte.to(tl.uint32) << 23).to(
                tl.float32, bitcast=True
            )
        else:
            scale = tl.load(cache1_f32 + scale_base // 4
                            + dims // _QUANT_BLOCK_SIZE,
                            mask=valid & is_nope, other=0.0)
        if CACHE1_FP8 or CACHE1_OFFICIAL_FP8:
            nope = tl.load((cache1_i8 + data_base + dims).to(
                tl.pointer_type(tl.float8e4nv)), mask=dim_mask & is_nope)
        else:
            nope = tl.load(cache1_i8 + data_base + dims,
                           mask=dim_mask & is_nope)
        rope = tl.load(cache1_bf16 + data_base // 2 + _ROPE_BF16_OFFSET
                       + dims - _NOPE_DIM,
                       mask=dim_mask & ~is_nope)
        value = tl.where(is_nope, nope.to(tl.float32) * scale, rope)
        output_item = length0 + item
        out_base = (row * out_width + output_item) * _HEAD_DIM
        tl.store(out + out_base + dims, value, mask=dim_mask)
        tl.store(out_indices + row * out_width + output_item,
                 row * out_width + output_item, mask=valid)
        tl.store(out_indices + row * out_width + output_item,
                 -1, mask=~valid)


def gather_two_int8_cache_indices(
    cache0: torch.Tensor,
    indices0: torch.Tensor,
    lengths0: torch.Tensor,
    cache1: torch.Tensor,
    indices1: torch.Tensor,
    lengths1: torch.Tensor,
    out: torch.Tensor,
    out_indices: torch.Tensor,
    cache0_fp8: bool = False,
    cache1_fp8: bool = False,
    cache1_official_fp8: bool = False,
) -> None:
    """Gather two cache pools into one packed BF16 workspace."""
    rows = indices0.shape[0]
    width0, width1 = indices0.shape[-1], indices1.shape[-1]
    flat0, flat1 = cache0.view(torch.uint8), cache1.view(torch.uint8)
    workers = min(max(width0, width1), 768)
    _gather_two_int8_indices_kernel[(rows, workers)](
        flat0.view(torch.int8), flat0.view(torch.bfloat16),
        flat0.view(torch.float32), indices0, lengths0, indices0.stride(0),
        flat1.view(torch.int8), flat1.view(torch.bfloat16),
        flat1.view(torch.float32), indices1, lengths1, indices1.stride(0),
        out, out_indices, width0, width1, out.shape[1],
        cache0.shape[1], cache0.stride(0),
        cache1.shape[1], cache1.stride(0),
        BLOCK=512, CACHE0_FP8=cache0_fp8, CACHE1_FP8=cache1_fp8,
        CACHE1_OFFICIAL_FP8=cache1_official_fp8,
        num_warps=2,
    )


@triton.jit
def _int8_sparse_decode_split_kernel(
    q, q_stride_r, q_stride_h,
    c0_i8, c0_bf16, c0_f32, i0, i0_stride, l0,
    c1_i8, c1_bf16, c1_f32, i1, i1_stride, l1,
    partial, p_stride_s, p_stride_r, p_stride_h,
    c0_bs, c0_stride, c1_bs, c1_stride,
    softmax_scale: tl.constexpr, NUM_HEADS: tl.constexpr,
    WIDTH0: tl.constexpr, WIDTH1: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Direct INT8-cache sparse MLA partials; no BF16 gather workspace."""
    pid = tl.program_id(0)
    head_blocks: tl.constexpr = tl.cdiv(NUM_HEADS, BLOCK_H)
    split = pid % NUM_SPLITS
    owner = pid // NUM_SPLITS
    row = (owner // head_blocks).to(tl.int64)
    h0 = owner % head_blocks * BLOCK_H
    heads = h0 + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, 256)
    items = tl.arange(0, BLOCK_K)

    q_ptr = (q + row * q_stride_r
             + heads[:, None] * q_stride_h + dims[None, :])
    q0 = tl.load(q_ptr, mask=heads[:, None] < NUM_HEADS, other=0.0)
    q1 = tl.load(q_ptr + 256,
                 mask=heads[:, None] < NUM_HEADS, other=0.0)
    len0 = tl.load(l0 + row)
    len1 = tl.load(l1 + row)
    total_len = len0 + len1

    max_value = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    acc0 = tl.zeros((BLOCK_H, 256), tl.float32)
    acc1 = tl.zeros((BLOCK_H, 256), tl.float32)
    num_tiles: tl.constexpr = (WIDTH0 + WIDTH1 + BLOCK_K - 1) // BLOCK_K
    tiles_per_split: tl.constexpr = (
        num_tiles + NUM_SPLITS - 1
    ) // NUM_SPLITS

    for local_tile in range(0, tiles_per_split):
        tile = split * tiles_per_split + local_tile
        logical = tile * BLOCK_K + items
        from0 = logical < len0
        from1 = (logical >= len0) & (logical < total_len)
        pos0 = tl.minimum(logical, tl.maximum(len0 - 1, 0))
        pos1 = tl.minimum(
            tl.maximum(logical - len0, 0), tl.maximum(len1 - 1, 0)
        )
        slot0 = tl.load(
            i0 + row * i0_stride + pos0, mask=from0, other=0
        ).to(tl.int64)
        slot1 = tl.load(
            i1 + row * i1_stride + pos1, mask=from1, other=0
        ).to(tl.int64)
        block0, offset0 = slot0 // c0_bs, slot0 % c0_bs
        block1, offset1 = slot1 // c1_bs, slot1 % c1_bs
        base0 = block0 * c0_stride + offset0 * _TOKEN_DATA_BYTES
        base1 = block1 * c1_stride + offset1 * _TOKEN_DATA_BYTES
        scale_ptr0 = (block0 * c0_stride
                      + c0_bs * _TOKEN_DATA_BYTES
                      + offset0 * _SCALE_SLOT_BYTES)
        scale_ptr1 = (block1 * c1_stride
                      + c1_bs * _TOKEN_DATA_BYTES
                      + offset1 * _SCALE_SLOT_BYTES)
        scale0_low = tl.load(
            c0_f32 + scale_ptr0[None, :] // 4
            + dims[:, None] // _QUANT_BLOCK_SIZE,
            mask=from0[None, :], other=0.0,
        )
        scale1_low = tl.load(
            c1_f32 + scale_ptr1[None, :] // 4
            + dims[:, None] // _QUANT_BLOCK_SIZE,
            mask=from1[None, :], other=0.0,
        )

        low0 = tl.load(
            c0_i8 + base0[None, :] + dims[:, None],
            mask=from0[None, :], other=0,
        ).to(tl.float32) * scale0_low
        low1 = tl.load(
            c1_i8 + base1[None, :] + dims[:, None],
            mask=from1[None, :], other=0,
        ).to(tl.float32) * scale1_low
        kv0 = (low0 + low1).to(tl.bfloat16)

        upper_dims = dims + 256
        is_nope = upper_dims < _NOPE_DIM
        scale0_high = tl.load(
            c0_f32 + scale_ptr0[None, :] // 4
            + upper_dims[:, None] // _QUANT_BLOCK_SIZE,
            mask=from0[None, :] & is_nope[:, None], other=0.0,
        )
        scale1_high = tl.load(
            c1_f32 + scale_ptr1[None, :] // 4
            + upper_dims[:, None] // _QUANT_BLOCK_SIZE,
            mask=from1[None, :] & is_nope[:, None], other=0.0,
        )
        nope0 = tl.load(
            c0_i8 + base0[None, :] + upper_dims[:, None],
            mask=from0[None, :] & is_nope[:, None], other=0,
        ).to(tl.float32) * scale0_high
        nope1 = tl.load(
            c1_i8 + base1[None, :] + upper_dims[:, None],
            mask=from1[None, :] & is_nope[:, None], other=0,
        ).to(tl.float32) * scale1_high
        rope_dims = upper_dims - _NOPE_DIM
        rope0 = tl.load(
            c0_bf16 + base0[None, :] // 2 + _ROPE_BF16_OFFSET
            + rope_dims[:, None],
            mask=from0[None, :] & ~is_nope[:, None], other=0.0,
        )
        rope1 = tl.load(
            c1_bf16 + base1[None, :] // 2 + _ROPE_BF16_OFFSET
            + rope_dims[:, None],
            mask=from1[None, :] & ~is_nope[:, None], other=0.0,
        )
        kv1 = tl.where(
            is_nope[:, None], nope0 + nope1, rope0 + rope1
        ).to(tl.bfloat16)

        scores = tl.dot(q0, kv0, out_dtype=tl.float32)
        scores = tl.dot(q1, kv1, scores,
                        out_dtype=tl.float32) * softmax_scale
        valid = (tile < num_tiles) & (logical < total_len)
        scores = tl.where(valid[None, :], scores, float("-inf"))
        new_max = tl.maximum(max_value, tl.max(scores, axis=1))
        # CUDA graph padding can produce rows (and split partitions) with no
        # valid cache entries.  Guard -inf - -inf explicitly: allowing that
        # NaN into the online-softmax state contaminates the final output even
        # though every lane is masked.
        has_new_value = new_max != float("-inf")
        alpha = tl.where(has_new_value, tl.exp(max_value - new_max), 0.0)
        probabilities = tl.where(
            valid[None, :], tl.exp(scores - new_max[:, None]), 0.0
        )
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        acc0 = tl.dot(
            probabilities.to(tl.bfloat16), tl.trans(kv0),
            acc0 * alpha[:, None], out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            probabilities.to(tl.bfloat16), tl.trans(kv1),
            acc1 * alpha[:, None], out_dtype=tl.float32,
        )
        max_value = new_max

    partial_head = (partial + split * p_stride_s + row * p_stride_r
                    + heads * p_stride_h)
    partial_ptr = partial_head[:, None] + dims[None, :]
    head_mask = heads[:, None] < NUM_HEADS
    tl.store(partial_ptr, acc0, mask=head_mask)
    tl.store(partial_ptr + 256, acc1, mask=head_mask)
    tl.store(partial_head + 512, max_value, mask=heads < NUM_HEADS)
    tl.store(partial_head + 513, denominator, mask=heads < NUM_HEADS)


@triton.jit
def _int8_sparse_decode_combine_kernel(
    partial, p_stride_s, p_stride_r, p_stride_h,
    output, out_stride_r, out_stride_h, attn_sink,
    NUM_HEADS: tl.constexpr, BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    head_blocks: tl.constexpr = tl.cdiv(NUM_HEADS, BLOCK_H)
    row = (pid // head_blocks).to(tl.int64)
    h0 = pid % head_blocks * BLOCK_H
    heads = h0 + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, 256)
    head_mask = heads[:, None] < NUM_HEADS
    global_max = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        ptr = (partial + split * p_stride_s + row * p_stride_r
               + heads * p_stride_h)
        global_max = tl.maximum(
            global_max,
            tl.load(ptr + 512, mask=heads < NUM_HEADS,
                    other=float("-inf")),
        )
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    acc0 = tl.zeros((BLOCK_H, 256), tl.float32)
    acc1 = tl.zeros((BLOCK_H, 256), tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        ptr = (partial + split * p_stride_s + row * p_stride_r
               + heads * p_stride_h)
        partial_max = tl.load(
            ptr + 512, mask=heads < NUM_HEADS, other=float("-inf")
        )
        alpha = tl.where(
            partial_max != float("-inf"),
            tl.exp(partial_max - global_max),
            0.0,
        )
        denominator += tl.load(
            ptr + 513, mask=heads < NUM_HEADS, other=0.0
        ) * alpha
        values = ptr[:, None] + dims[None, :]
        acc0 += tl.load(values, mask=head_mask, other=0.0) * alpha[:, None]
        acc1 += tl.load(values + 256,
                        mask=head_mask, other=0.0) * alpha[:, None]
    sink = tl.load(attn_sink + heads, mask=heads < NUM_HEADS,
                   other=float("-inf"))
    has_value = global_max != float("-inf")
    factor = tl.where(
        has_value,
        1.0 / (denominator + tl.exp(sink - global_max)),
        0.0,
    )
    out_ptr = (output + row * out_stride_r
               + heads[:, None] * out_stride_h + dims[None, :])
    tl.store(out_ptr, (acc0 * factor[:, None]).to(tl.bfloat16),
             mask=head_mask)
    tl.store(out_ptr + 256, (acc1 * factor[:, None]).to(tl.bfloat16),
             mask=head_mask)


def int8_sparse_decode_attention(
    q: torch.Tensor,
    cache0: torch.Tensor,
    indices0: torch.Tensor,
    lengths0: torch.Tensor,
    cache1: torch.Tensor,
    indices1: torch.Tensor,
    lengths1: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    output: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    """Attend directly to two packed INT8 caches without an FP8/BF16 cache."""
    block_h, block_k, num_splits = 16, 32, 4
    assert q.shape[-1] == _HEAD_DIM.value
    assert partial.shape == (num_splits, q.shape[0], q.shape[1], 514)
    flat0 = cache0.view(torch.uint8)
    flat1 = cache1.view(torch.uint8)
    head_blocks = triton.cdiv(q.shape[1], block_h)
    _int8_sparse_decode_split_kernel[
        (q.shape[0] * head_blocks * num_splits,)
    ](
        q, q.stride(0), q.stride(1),
        flat0.view(torch.int8), flat0.view(torch.bfloat16),
        flat0.view(torch.float32), indices0, indices0.stride(0), lengths0,
        flat1.view(torch.int8), flat1.view(torch.bfloat16),
        flat1.view(torch.float32), indices1, indices1.stride(0), lengths1,
        partial, partial.stride(0), partial.stride(1), partial.stride(2),
        cache0.shape[1], cache0.stride(0),
        cache1.shape[1], cache1.stride(0),
        softmax_scale=softmax_scale, NUM_HEADS=q.shape[1],
        WIDTH0=indices0.shape[1], WIDTH1=indices1.shape[1],
        BLOCK_H=block_h, BLOCK_K=block_k, NUM_SPLITS=num_splits,
        num_warps=8, num_stages=3,
    )
    _int8_sparse_decode_combine_kernel[(q.shape[0] * head_blocks,)](
        partial, partial.stride(0), partial.stride(1), partial.stride(2),
        output, output.stride(0), output.stride(1), attn_sink,
        NUM_HEADS=q.shape[1], BLOCK_H=block_h, NUM_SPLITS=num_splits,
        num_warps=8,
    )


@triton.jit
def _gather_int8_paged_kernel(
    cache_i8,
    cache_bf16,
    cache_f32,
    out,
    seq_lens,
    gather_lens,
    block_table,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    cache_block_stride: tl.constexpr,
    offset,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    has_gather_lens: tl.constexpr,
    BLOCK: tl.constexpr, CACHE_FP8: tl.constexpr,
    OFFICIAL_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    worker = tl.program_id(1)
    num_workers = tl.num_programs(1)
    seq_len = tl.load(seq_lens + row)
    gather_len = (
        tl.load(gather_lens + row) if has_gather_lens else seq_len
    )
    start = seq_len - gather_len
    dims = tl.arange(0, BLOCK)
    for token in range(worker, gather_len, num_workers):
        logical_token = start + token
        logical_block = logical_token // block_size
        block_offset = logical_token % block_size
        physical_block = tl.load(
            block_table + row * block_table_stride + logical_block,
        ).to(tl.int64)
        block_base = physical_block * cache_block_stride
        data_base = block_base + block_offset * _TOKEN_DATA_BYTES
        scale_base = (
            block_base
            + block_size * _TOKEN_DATA_BYTES
            + block_offset * _SCALE_SLOT_BYTES
        )
        is_nope = dims < _NOPE_DIM
        if OFFICIAL_FP8:
            official_scale_base = (
                block_base + block_size * _TOKEN_DATA_BYTES
                + block_offset * 8
            )
            scale_byte = tl.load(
                cache_i8.to(tl.pointer_type(tl.uint8))
                + official_scale_base + dims // _QUANT_BLOCK_SIZE,
                mask=is_nope, other=0,
            )
            scale = (scale_byte.to(tl.uint32) << 23).to(
                tl.float32, bitcast=True
            )
        else:
            scale = tl.load(cache_f32 + scale_base // 4
                            + dims // _QUANT_BLOCK_SIZE,
                            mask=is_nope, other=0.0)
        if CACHE_FP8 or OFFICIAL_FP8:
            nope = tl.load(
                (cache_i8 + data_base + dims).to(
                    tl.pointer_type(tl.float8e4nv)
                ), mask=is_nope, other=0.0,
            )
        else:
            nope = tl.load(
                cache_i8 + data_base + dims, mask=is_nope, other=0,
            )
        rope = tl.load(
            cache_bf16 + data_base // 2 + _ROPE_BF16_OFFSET + (dims - _NOPE_DIM),
            mask=~is_nope,
            other=0,
        )
        value = tl.where(is_nope, nope.to(tl.float32) * scale, rope)
        out_base = row * out_stride0 + (offset + token) * out_stride1
        tl.store(out + out_base + dims, value)


def dequantize_and_gather_int8_paged_cache(
    out: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    fp8_cache: bool = False,
    official_fp8_cache: bool = False,
) -> None:
    """Paged-cache equivalent of DeepSeek's FP8 gather for INT8 caches."""
    rows = seq_lens.shape[0]
    if rows == 0 or out.shape[1] == offset:
        return
    flat = cache.view(torch.uint8)
    num_workers = 128
    _gather_int8_paged_kernel[(rows, num_workers)](
        flat.view(torch.int8),
        flat.view(torch.bfloat16),
        flat.view(torch.float32),
        out,
        seq_lens,
        gather_lens,
        block_table,
        block_table.stride(0),
        block_size,
        cache.stride(0),
        offset,
        out.stride(0),
        out.stride(1),
        has_gather_lens=gather_lens is not None,
        BLOCK=512, CACHE_FP8=fp8_cache,
        OFFICIAL_FP8=official_fp8_cache,
        num_warps=4,
    )
