# SPDX-License-Identifier: Apache-2.0
"""Runtime installation of the direct packed-INT8 sparse decode path."""

import os
import logging

import torch
import torch.nn.functional as F

from vllm.v1.worker.workspace import current_workspace_manager
from vllm.triton_utils import tl, triton
from vllm_fl.ops import deepseek_v4_int8_kv as _base

_NOPE_DIM = _base._NOPE_DIM
_HEAD_DIM = _base._HEAD_DIM
_TOKEN_DATA_BYTES = _base._TOKEN_DATA_BYTES
_QUANT_BLOCK_SIZE = _base._QUANT_BLOCK_SIZE
_SCALE_SLOT_BYTES = _base._SCALE_SLOT_BYTES
_ROPE_BF16_OFFSET = _base._ROPE_BF16_OFFSET

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
    # Keep the sparse-width loop dynamic. Statically unrolling the production
    # top-k width duplicates the two 256-wide dot products dozens of times and
    # makes every TP rank spend minutes compiling the same giant kernel.
    tile = split
    while tile * BLOCK_K < total_len:
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
        valid = logical < total_len
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
        tile += NUM_SPLITS

    partial_head = (partial + split * p_stride_s + row * p_stride_r
                    + heads * p_stride_h)
    partial_ptr = partial_head[:, None] + dims[None, :]
    head_mask = heads[:, None] < NUM_HEADS
    tl.store(partial_ptr, acc0, mask=head_mask)
    tl.store(partial_ptr + 256, acc1, mask=head_mask)
    tl.store(partial_head + 512, max_value, mask=heads < NUM_HEADS)
    tl.store(partial_head + 513, denominator, mask=heads < NUM_HEADS)


int8_sparse_decode_attention = _base.int8_sparse_decode_attention

_ORIGINAL_FORWARD_DECODE = None
_ENABLED = os.getenv("VLLM_FL_INT8_DIRECT_DECODE", "1").lower() not in (
    "0", "false", "off", "no"
)


def _forward_decode_direct_int8(
    self,
    q,
    kv_cache,
    swa_metadata,
    attn_metadata,
    swa_only,
    output,
):
    if not _ENABLED or not self.use_int8_kv_cache:
        return _ORIGINAL_FORWARD_DECODE(
            self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
        )

    num_decodes = swa_metadata.num_decodes
    num_decode_tokens = swa_metadata.num_decode_tokens
    if q.shape[1] < self.padded_heads:
        q = F.pad(q, (0, 0, 0, self.padded_heads - q.shape[1]), value=0.0)

    topk_indices = None
    topk_lens = None
    if not swa_only:
        assert attn_metadata is not None
        assert swa_metadata.is_valid_token is not None
        block_size = attn_metadata.block_size // self.compress_ratio
        is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
        if self.compress_ratio == 4:
            from vllm_fl.ops.deepseek_v4_attention import (
                _compute_global_topk_indices_and_lens,
            )

            assert self.topk_indices_buffer is not None
            global_indices, topk_lens = _compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            topk_indices = global_indices.view(num_decode_tokens, 1, -1)
        else:
            topk_indices = attn_metadata.c128a_global_decode_topk_indices
            topk_lens = attn_metadata.c128a_decode_topk_lens

    swa_indices = swa_metadata.decode_swa_indices
    swa_lens = swa_metadata.decode_swa_lens
    main_indices = (
        swa_indices[..., :0] if topk_indices is None else topk_indices.squeeze(1)
    )
    main_lens = (
        torch.zeros_like(swa_lens) if topk_lens is None else topk_lens
    )
    cache0 = kv_cache if kv_cache is not None else self.swa_cache_layer.kv_cache
    (partial,) = current_workspace_manager().get_simultaneous(
        ((4, num_decode_tokens, self.padded_heads, 514), torch.float32),
    )
    int8_sparse_decode_attention(
        q,
        cache0,
        main_indices,
        main_lens,
        self.swa_cache_layer.kv_cache,
        swa_indices,
        swa_lens,
        self.attn_sink,
        self.scale,
        output,
        partial,
    )


def install_int8_direct_decode() -> bool:
    """Patch the runtime-only MLA implementation without changing its graph API."""
    global _ORIGINAL_FORWARD_DECODE
    from vllm_fl.ops.deepseek_v4_attention import DeepseekV4MLAAttention

    if getattr(DeepseekV4MLAAttention, "_fl_direct_int8_decode", False):
        return False
    _base._int8_sparse_decode_split_kernel = (
        _int8_sparse_decode_split_kernel
    )
    _ORIGINAL_FORWARD_DECODE = DeepseekV4MLAAttention._forward_decode
    DeepseekV4MLAAttention._forward_decode = _forward_decode_direct_int8
    DeepseekV4MLAAttention._fl_direct_int8_decode = True
    logging.getLogger(__name__).warning(
        "Enabled direct packed-INT8 sparse decode attention"
    )
    return True
