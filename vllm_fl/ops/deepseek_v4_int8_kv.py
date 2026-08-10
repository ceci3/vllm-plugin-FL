# SPDX-License-Identifier: Apache-2.0
"""Triton helpers for DeepSeek V4 per-token-head INT8 KV caches."""

import torch

from vllm.triton_utils import tl, triton

# DeepSeek-V4 INT8 page layout, shared by the writers and readers below.
# Declared as ``tl.constexpr`` so the @triton.jit kernels can reference them
# directly (plain module-level ints are rejected by Triton); use ``.value`` for
# host-side arithmetic.
#
# Per-token page (584B):
#   [0, 448)      448 x int8   NoPE
#   [448, 576)    64 x bfloat16 RoPE
#   [576, 584)    1 x float32 scale + 4B pad   (in the per-block scale region)
_NOPE_DIM = tl.constexpr(448)
_HEAD_DIM = tl.constexpr(512)
_TOKEN_DATA_BYTES = tl.constexpr(576)
_SCALE_SLOT_BYTES = tl.constexpr(8)
# Offset of the RoPE half when the page is viewed as bfloat16 (448 / 2).
_ROPE_BF16_OFFSET = tl.constexpr(224)

# Plain-int mirror for host-side callers (KV cache shape/spec construction),
# which cannot index with a tl.constexpr.
INT8_TOKEN_PAGE_BYTES = 584


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
    KV_BLOCK_STRIDE: tl.constexpr,
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
    absmax = tl.max(tl.where(nope_mask, tl.abs(normalized), 0.0), axis=0)
    scale = tl.maximum(absmax / 127.0, 1.0e-12)
    quant = _round_to_int8(normalized / scale)
    kv_block = kv_slot // kv_block_size
    kv_pos = kv_slot % kv_block_size
    block_base = kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    data_base = block_base + kv_pos * TOKEN_STRIDE
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
    tl.store(k_cache.to(tl.pointer_type(tl.float32)) + scale_base // 4, scale)


@triton.jit
def _qnorm_rope_kv_insert_int8_mla_kernel(
    q, q_stride_t, q_stride_h, kv, kv_stride_t, cache, slots, positions,
    cos_sin, cos_sin_stride, eps, block_size, cache_block_stride,
    NUM_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr, FLAT_GRID: tl.constexpr,
):
    """Fuse SWA query preparation and direct INT8 KV insertion.

    The second grid dimension selects a query head, with one additional
    program for the shared KV head.  This halves launch count without
    coupling the INT8 cache layout to the FP8 implementation.
    """
    q_tasks_per_token: tl.constexpr = NUM_HEADS
    tasks_per_token: tl.constexpr = q_tasks_per_token + 1
    if FLAT_GRID:
        pid = tl.program_id(0).to(tl.int64)
        token = pid // tasks_per_token
        task = pid % tasks_per_token
    else:
        token = tl.program_id(0).to(tl.int64)
        task = tl.program_id(1).to(tl.int64)
    is_kv = task == q_tasks_per_token
    dims = tl.arange(0, HEAD_SIZE)
    nope_dim: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    rope_dims = tl.arange(0, ROPE_HEAD_DIM)
    half_rope_dim: tl.constexpr = ROPE_HEAD_DIM // 2
    half_rope_dims = tl.arange(0, half_rope_dim)
    position = tl.load(positions + token)
    cs = cos_sin + position * cos_sin_stride
    cos_v = tl.load(cs + half_rope_dims)
    sin_v = tl.load(cs + half_rope_dim + half_rope_dims)

    if not is_kv:
        ptr = q + token * q_stride_t + task * q_stride_h
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
    else:
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
            absmax = tl.max(tl.where(nope_mask, tl.abs(x), 0.0), axis=0)
            scale = tl.maximum(absmax / 127.0, 1.0e-12)
            quant = _round_to_int8(x / scale)
            block = slot // block_size
            pos = slot % block_size
            block_base = block * cache_block_stride
            data_base = block_base + pos * _TOKEN_DATA_BYTES
            tl.store(cache + data_base + dims, quant, mask=nope_mask)
            rope_ptr = (cache + data_base + nope_dim).to(
                tl.pointer_type(tl.bfloat16))
            tl.store(rope_ptr + rope_dims, kv_rotated)
            scale_base = (
                block_base
                + block_size * _TOKEN_DATA_BYTES
                + pos * _SCALE_SLOT_BYTES
            )
            tl.store(cache.to(tl.pointer_type(tl.float32)) + scale_base // 4,
                     scale)


def qnorm_rope_kv_insert_int8_mla(
    q: torch.Tensor, kv: torch.Tensor, cache: torch.Tensor,
    slots: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
    eps: float, block_size: int,
) -> None:
    """Apply Q RMSNorm/RoPE and write SWA KV directly as INT8."""
    pos = positions.to(torch.int64)
    flat = cache.view(torch.uint8)
    grid = (q.shape[0], q.shape[1] + 1)
    _qnorm_rope_kv_insert_int8_mla_kernel[grid](
        q, q.stride(0), q.stride(1), kv, kv.stride(0), flat, slots, pos,
        cos_sin, cos_sin.stride(0), eps, block_size, cache.stride(0),
        NUM_HEADS=q.shape[1], HEAD_SIZE=q.shape[2], ROPE_HEAD_DIM=64,
        FLAT_GRID=False,
        num_warps=1,
        num_stages=2,
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
    out_width,
    cache_block_size,
    cache_block_stride,
    has_item_offsets: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    item = tl.program_id(1)

    valid = item < tl.load(lengths + row)
    slot = tl.load(
        indices + row * indices_stride + item, mask=valid, other=0
    ).to(tl.int64)

    dims = tl.arange(0, BLOCK)
    dim_mask = valid & (dims < _HEAD_DIM)
    cache_block = slot // cache_block_size
    cache_pos = slot % cache_block_size
    block_base = cache_block * cache_block_stride
    data_base = block_base + cache_pos * _TOKEN_DATA_BYTES
    scale_base = (
        block_base
        + cache_block_size * _TOKEN_DATA_BYTES
        + cache_pos * _SCALE_SLOT_BYTES
    )
    scale = tl.load(
        cache_f32 + scale_base // 4,
        mask=valid,
        other=0.0,
    )
    is_nope = dims < _NOPE_DIM
    nope = tl.load(cache_i8 + data_base + dims, mask=dim_mask & is_nope)
    rope = tl.load(
        cache_bf16 + data_base // 2 + _ROPE_BF16_OFFSET + (dims - _NOPE_DIM),
        mask=dim_mask & ~is_nope,
    )
    value = tl.where(is_nope, nope.to(tl.float32) * scale, rope)
    row_item_offset = (
        tl.load(item_offsets + row) if has_item_offsets else item_offset
    )
    output_item = row_item_offset + item
    out_base = (row * out_width + output_item) * _HEAD_DIM
    tl.store(out + out_base + dims, value, mask=dim_mask)
    tl.store(
        out_indices + row * out_width + output_item,
        row * out_width + output_item,
        mask=valid,
    )
    tl.store(
        out_indices + row * out_width + output_item,
        -1,
        mask=~valid,
    )


def gather_int8_cache_indices(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    out_indices: torch.Tensor,
    item_offset: int = 0,
    item_offsets: torch.Tensor | None = None,
) -> None:
    """Gather sparse slots and dequantize them into a dense workspace."""
    rows, width = indices.shape[0], indices.shape[-1]
    flat = cache.view(torch.uint8)
    _gather_int8_indices_kernel[(rows, width)](
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
        out.shape[1],
        cache.shape[1],
        cache.stride(0),
        has_item_offsets=item_offsets is not None,
        BLOCK=512,
        num_warps=4,
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
    BLOCK: tl.constexpr,
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
        scale = tl.load(cache_f32 + scale_base // 4)
        is_nope = dims < _NOPE_DIM
        nope = tl.load(
            cache_i8 + data_base + dims,
            mask=is_nope,
            other=0,
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
        BLOCK=512,
        num_warps=4,
    )
