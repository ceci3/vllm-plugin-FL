# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
import json
import os
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm_fl.ops.deepseek_v4_int8_kv import (
    fused_compress_rope_int8_mla_cache_kernel,
)
from vllm_fl.ops.deepseek_v4_int8_indexer import (
    fused_compress_rope_int8_indexer_cache_kernel,
)
from vllm_fl.dispatch.backends.vendor.cuda.impl.deepseek_v4_ops.fused_compress import (
    _fused_kv_compress_norm_rope_insert_sparse_attn_bf16,
    _fused_kv_compress_norm_rope_insert_indexer_attn_bf16,
)

class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 608] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.quant_config = vllm_config.quant_config
        self.use_fp4_cache = use_fp4_cache
        requested_int8 = (
            vllm_config.cache_config.cache_dtype
            == "int8_per_token_head"
        )
        requested_bf16 = vllm_config.cache_config.cache_dtype in (
            "bf16", "bfloat16"
        )
        layer_list = os.getenv("VLLM_DSV4_INT8_ATTN_LAYERS")
        layer_range = os.getenv("VLLM_DSV4_INT8_ATTN_LAYER_RANGE")
        if requested_int8 and layer_list and ".layers." in prefix:
            layer = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
            requested_int8 = layer in {
                int(item) for item in layer_list.split(",") if item.strip()
            }
        elif requested_int8 and layer_range and ".layers." in prefix:
            start_text, end_text = layer_range.split(":", 1)
            layer = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
            requested_int8 = int(start_text) <= layer <= int(end_text)
        indexer_override = os.getenv("VLLM_DSV4_INT8_INDEXER_CACHE")
        if head_dim == 128 and indexer_override is not None:
            requested_int8 = indexer_override == "1"
        layer_mode = os.getenv("VLLM_DSV4_INT8_ATTN_LAYER_MODE")
        if head_dim == 512:
            layer_is_compressed = compress_ratio > 1
            if layer_mode == "swa_only" and layer_is_compressed:
                requested_int8 = False
            elif layer_mode == "compressed" and not layer_is_compressed:
                requested_int8 = False
        self.use_int8_kv_cache = head_dim == 512 and requested_int8
        self.use_int8_indexer_cache = head_dim == 128 and requested_int8
        skip_layers = {
            int(value) for value in os.getenv(
                "VLLM_DSV4_INT8_KV_SKIP_LAYERS", ""
            ).split(",") if value.strip()
        }
        current_layer = (
            int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
            if ".layers." in prefix else -1
        )
        self.store_fp8_main_cache = (
            self.use_int8_kv_cache and current_layer in skip_layers
        )
        self._quant_diag_done = False

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        # NOTE: the cache element format is chosen by the *cache* dtype, not by
        # the weight quantization config — an INT-quantized checkpoint can still
        # hold an FP8, MXFP4 or INT8 KV cache. The writer picked here must match
        # what the reader in deepseek_v4_attention.py expects, otherwise the
        # cached bytes get reinterpreted under the wrong encoding.
        if self.quant_config is not None and not requested_bf16:
            if self.head_dim == 512:
                assert not use_fp4_cache, (
                    "MXFP4 cache is only supported for indexer (head=128)"
                )
                # INT8 shares the FP8 page geometry (576B data + 8B scale slot);
                # only the NoPE encoding and the scale semantics differ.
                self._fused_kernel = (
                    fused_compress_rope_int8_mla_cache_kernel
                    if self.use_int8_kv_cache
                    else _fused_kv_compress_norm_rope_insert_sparse_attn
                )
                self._quant_block = 64
                self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
                self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
                self._num_warps = 4
            elif self.head_dim == 128:
                if self.use_int8_indexer_cache:
                    self._fused_kernel = (
                        fused_compress_rope_int8_indexer_cache_kernel
                    )
                    self._quant_block = 128
                    self._token_stride = self.head_dim
                    self._scale_dim = 4  # single float32 scale
                elif use_fp4_cache:
                    self._fused_kernel = (
                        _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
                    )
                    self._quant_block = MXFP4_BLOCK_SIZE
                    self._token_stride = self.head_dim // 2
                    self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
                else:
                    self._fused_kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
                    self._quant_block = 128
                    self._token_stride = self.head_dim
                    self._scale_dim = 4  # single float32 scale
                self._num_warps = 1
            else:
                raise ValueError(
                    f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
                )
        else:
            ### USE BF16 KERNELS
            if self.head_dim == 512:
                self._fused_kernel = _fused_kv_compress_norm_rope_insert_sparse_attn_bf16
                # Triton pointer arithmetic is in elements, not bytes.  The
                # BF16 writer receives a bfloat16* cache pointer, so advancing
                # one token requires HEAD_SIZE elements (512), not 1024 bytes.
                self._token_stride = self.head_dim
                self._num_warps = 4
            elif self.head_dim == 128:
                self._fused_kernel = _fused_kv_compress_norm_rope_insert_indexer_attn_bf16
                self._token_stride = self.head_dim
                self._num_warps = 1
            else:
                raise ValueError(
                    f"Unsupported head_dim for fused bf16 cache: {self.head_dim}"
                )
            self._scale_dim = None
            self._quant_block = None

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and _fused_kernel below
        # depend on preceding kernel outputs (kv/score from the cublas GEMM;
        # state_cache from this kernel) but neither emits/waits on PDL grid
        # dependency primitives, so launch_pdl=True caused a read-after-write
        # race and non-deterministic output.
        _save_partial_states_kernel[(num_actual,)](
            kv,
            kv.stride(0),
            score,
            score.stride(0),
            self.ape,
            self.ape.stride(0),
            positions,
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            slot_mapping,
            block_size,
            HEAD_SIZE=kv.shape[-1],
            TRITON_BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            launch_pdl=False,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

        quant_kwargs = {}
        if self._scale_dim is not None:
            quant_kwargs = {
                "FP8_MAX": 448.0,
                "QUANT_BLOCK": self._quant_block,
                "SCALE_DIM": self._scale_dim,
            }
            if self.head_dim == 512 and self.use_int8_kv_cache:
                quant_kwargs["STORE_FP8"] = self.store_fp8_main_cache

        self._fused_kernel[(num_actual,)](
            # state cache
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            # metadata
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            block_size,
            # RMSNorm
            self.norm.weight,
            self.rms_norm_eps,
            # RoPE
            cos_sin_cache,
            cos_sin_cache.stride(0),
            # KV cache
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size (tokens per block)
            # constexprs
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            ROPE_HEAD_DIM=self.rope_head_dim,
            TOKEN_STRIDE=self._token_stride,
            KV_BLOCK_STRIDE=kv_cache.stride(0),
            num_warps=self._num_warps,
            launch_pdl=False,
            **quant_kwargs,
        )

        if (
            os.getenv("VLLM_DSV4_MAIN_KV_QUANT_DIAG") == "1"
            and self.head_dim == 512
            and not self._quant_diag_done
            and (
                os.getenv("VLLM_DSV4_MAIN_KV_QUANT_DIAG_LAYER", "layers.2")
                == "all"
                or os.getenv(
                    "VLLM_DSV4_MAIN_KV_QUANT_DIAG_LAYER", "layers.2"
                ) in self.prefix
            )
        ):
            self._quant_diag_done = True
            self._diagnose_main_kv_quantization(
                state_cache, state_width, token_to_req_indices, positions,
                slot_mapping, block_table, block_size, kv_cache,
                k_cache_metadata.slot_mapping, cos_sin_cache,
            )

    @torch.no_grad()
    def _diagnose_main_kv_quantization(
        self, state_cache: torch.Tensor, state_width: int,
        token_to_req: torch.Tensor, positions: torch.Tensor,
        state_slots: torch.Tensor, block_table: torch.Tensor,
        state_block_size: int, kv_cache: torch.Tensor,
        kv_slots: torch.Tensor, cos_sin_cache: torch.Tensor,
    ) -> None:
        """Compare production per-head INT8 with hypothetical 7-block INT8."""
        valid = (
            (state_slots >= 0)
            & (((positions + 1) % self.compress_ratio) == 0)
        ).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            self._quant_diag_done = False
            return

        pos = positions[valid].long()
        req = token_to_req[valid].long()
        count = self.coff * self.compress_ratio
        history = torch.arange(count, device=pos.device, dtype=torch.long)
        history_pos = pos[:, None] - count + 1 + history[None, :]
        history_valid = history_pos >= 0
        safe_history_pos = history_pos.clamp_min(0)
        physical_blocks = block_table[
            req[:, None], safe_history_pos // state_block_size
        ].long()
        block_offsets = safe_history_pos % state_block_size
        rows = state_cache[physical_blocks, block_offsets]

        dims = torch.arange(self.head_dim, device=pos.device)
        overlap = (history >= self.compress_ratio).long() * self.head_dim
        value_index = overlap[None, :, None] + dims[None, None, :]
        value_index = value_index.expand(valid.numel(), -1, -1)
        values = torch.gather(rows, 2, value_index)
        scores = torch.gather(rows, 2, state_width + value_index)
        scores = scores.masked_fill(~history_valid[:, :, None], float("-inf"))
        compressed = (values * torch.softmax(scores, dim=1)).sum(dim=1)
        variance = compressed.square().mean(dim=-1, keepdim=True)
        normalized = (
            compressed * torch.rsqrt(variance + self.rms_norm_eps)
            * self.norm.weight.float()
        ).to(torch.bfloat16).float()
        x = normalized[:, :448]

        single_scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127
        single_dq = torch.round(x / single_scale).clamp(-127, 127) * single_scale
        x_blocks = x.view(-1, 7, 64)
        block_absmax = x_blocks.abs().amax(dim=-1, keepdim=True)
        block_scale = block_absmax.clamp_min(1e-12) / 127
        block_dq = (
            torch.round(x_blocks / block_scale).clamp(-127, 127) * block_scale
        ).view_as(x)
        fp8_raw_scale = block_absmax.clamp_min(1e-4) / 448.0
        fp8_scale = torch.pow(2.0, torch.ceil(torch.log2(fp8_raw_scale)))
        fp8_dq = (
            (x_blocks / fp8_scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .float()
            * fp8_scale
        ).view_as(x)

        # Per (compressed token, 64-channel group) diagnostics.  Keep the
        # highest-impact rows rather than dumping every activation value.
        block_error = (block_dq.view_as(x_blocks) - x_blocks).float()
        group_mae = block_error.abs().mean(dim=-1)
        group_rmse = block_error.square().mean(dim=-1).sqrt()
        group_max_error = block_error.abs().amax(dim=-1)
        group_rms = x_blocks.square().mean(dim=-1).sqrt()
        group_outlier_ratio = (
            block_absmax.squeeze(-1) / group_rms.clamp_min(1e-12)
        )

        def top_group_rows(metric: torch.Tensor, limit: int = 256) -> list[dict]:
            count = min(limit, metric.numel())
            flat_values, flat_indices = torch.topk(metric.flatten(), count)
            token_indices = flat_indices // 7
            group_indices = flat_indices % 7
            rows = []
            for value, token_index, group_index in zip(
                flat_values.cpu().tolist(),
                token_indices.cpu().tolist(),
                group_indices.cpu().tolist(),
            ):
                rows.append({
                    "metric": value,
                    "request_index": int(req[token_index].item()),
                    "source_position": int(pos[token_index].item()),
                    "compressed_position": int(
                        (pos[token_index] // self.compress_ratio).item()
                    ),
                    "group": int(group_index),
                    "channel_start": int(group_index * 64),
                    "absmax": float(
                        block_absmax[token_index, group_index, 0].item()
                    ),
                    "scale": float(block_scale[token_index, group_index, 0].item()),
                    "mae": float(group_mae[token_index, group_index].item()),
                    "rmse": float(group_rmse[token_index, group_index].item()),
                    "max_error": float(
                        group_max_error[token_index, group_index].item()
                    ),
                    "outlier_ratio": float(
                        group_outlier_ratio[token_index, group_index].item()
                    ),
                })
            return rows

        # Read back the exact bytes written by the production kernel. This
        # distinguishes quantization loss from writer history/slot/layout bugs.
        selected_kv_slots = kv_slots[valid].long()
        kv_block_size = kv_cache.shape[1]
        kv_blocks = selected_kv_slots // kv_block_size
        kv_offsets = selected_kv_slots % kv_block_size
        flat_u8 = kv_cache.view(torch.uint8).view(kv_cache.shape[0], -1)
        nope_bytes = (
            kv_offsets[:, None] * 576
            + torch.arange(448, device=x.device)[None, :]
        )
        writer_q = torch.gather(flat_u8[kv_blocks], 1, nope_bytes).view(torch.int8)
        flat_f32 = flat_u8.view(torch.float32)
        scale_indices = (
            (kv_block_size * 576 + kv_offsets[:, None] * 32) // 4
            + torch.arange(7, device=x.device)[None, :]
        )
        writer_scales = torch.gather(flat_f32[kv_blocks], 1, scale_indices)
        writer_dq = (
            writer_q.float().view(-1, 7, 64) * writer_scales[:, :, None]
        ).view_as(x)

        rope_byte_indices = (
            kv_offsets[:, None] * 576 + 448
            + torch.arange(128, device=x.device)[None, :]
        )
        writer_rope = torch.gather(
            flat_u8[kv_blocks], 1, rope_byte_indices
        ).contiguous().view(torch.bfloat16).float()
        rope = normalized[:, 448:].view(-1, 32, 2)
        compressed_pos = (pos // self.compress_ratio) * self.compress_ratio
        cs = cos_sin_cache[compressed_pos]
        cos, sin = cs[:, :32], cs[:, 32:64]
        rope_ref = torch.stack(
            (rope[..., 0] * cos - rope[..., 1] * sin,
             rope[..., 1] * cos + rope[..., 0] * sin),
            dim=-1,
        ).flatten(1).to(torch.bfloat16).float()

        def metrics(dq: torch.Tensor) -> dict[str, float]:
            error = dq - x
            signal = x.square().mean().sqrt()
            noise = error.square().mean().sqrt()
            cosine = torch.nn.functional.cosine_similarity(x, dq, dim=-1).mean()
            return {
                "mae": error.abs().mean().item(),
                "rmse": noise.item(),
                "max_abs": error.abs().amax().item(),
                "cosine": cosine.item(),
                "sqnr_db": (20 * torch.log10(signal / noise)).item(),
            }

        payload = {
            "prefix": self.prefix,
            "rank": int(os.getenv("LOCAL_RANK", "0")),
            "compress_ratio": self.compress_ratio,
            "tokens": valid.numel(),
            "single_scale": metrics(single_dq),
            "seven_block_scale": metrics(block_dq),
            "fp8_seven_block_ue8m0": metrics(fp8_dq),
            "writer_roundtrip": metrics(writer_dq),
            "writer_vs_seven_block_rmse": (
                (writer_dq - block_dq).square().mean().sqrt().item()
            ),
            "writer_scale_max_abs_diff": (
                writer_scales - block_scale.squeeze(-1)
            ).abs().amax().item(),
            "writer_rope_max_abs_diff": (
                writer_rope - rope_ref
            ).abs().amax().item(),
            "global_absmax_mean": x.abs().amax(dim=-1).mean().item(),
            "block_absmax_mean": block_absmax.mean(dim=(0, 2)).tolist(),
            "global_to_median_block_absmax": (
                x.abs().amax(dim=-1)
                / block_absmax.squeeze(-1).median(dim=-1).values.clamp_min(1e-12)
            ).mean().item(),
            "group_mae_mean": group_mae.mean(dim=0).tolist(),
            "group_rmse_mean": group_rmse.mean(dim=0).tolist(),
            "group_outlier_ratio_mean": group_outlier_ratio.mean(dim=0).tolist(),
            "top_quant_rmse": top_group_rows(group_rmse),
            "top_outlier_ratio": top_group_rows(group_outlier_ratio),
        }
        output_dir = os.getenv(
            "VLLM_DSV4_MAIN_KV_QUANT_DIAG_DIR", "/tmp/dsv4_main_kv_diag"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"rank{payload['rank']}_{self.prefix.replace('.', '_')}_"
            f"cr{self.compress_ratio}_{os.getpid()}.json",
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )
