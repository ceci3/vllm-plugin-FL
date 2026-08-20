"""Isolated sweep of unused TLE features on the DSV4 INT8 MQA kernel.

Baseline is the shipped kernel body (tle.load async + tl.range num_stages).
Each variant changes exactly one thing so the delta is attributable:

  base   : current production kernel body
  ws     : + tl.range(warp_specialize=True)
  reorder: + tle.range(loop_unroll_factor=2, reorder=True)
  wsre   : both

Correctness is checked against the production kernel before timing; a variant
that does not match bit-for-bit inside the key window is reported as WRONG and
its timing is meaningless.
"""

import argparse

import torch
from vllm.triton_utils import tl, triton

import triton.experimental.tle.language as tle
from vllm_fl.ops.deepseek_v4_int8_indexer import int8_mqa_logits


@triton.jit
def _n_tile(pid_n, q_tile, hw, lo, hi, row_mask, rows, k, ks, out, n,
            BM: tl.constexpr, BN: tl.constexpr):
    """One N tile: window test, async K load, Tensor Core dot, masked store."""
    dims = tl.arange(0, 128)
    keys = pid_n * BN + tl.arange(0, BN)
    key_mask = keys < n
    tmin = pid_n * BN
    tmax = tl.minimum(tmin + BN, n) - 1
    lo_min = tl.min(tl.where(row_mask, lo, n), axis=0)
    hi_max = tl.max(tl.where(row_mask, hi, 0), axis=0)
    if (tmax >= lo_min) & (tmin < hi_max):
        k_tile = tle.load(k + keys[:, None] * 128 + dims[None, :],
                          mask=key_mask[:, None], other=0, is_async=True)
        scales = tle.load(ks + keys, mask=key_mask, other=0.0, is_async=True)
        dots = tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.int32)
        scores = tl.reshape(dots, (BN, BM, 64)).to(tl.float32)
        vals = tl.sum(tl.maximum(scores * scales[:, None, None], 0.0)
                      * hw[None, :, :], axis=2)
        valid = (row_mask[:, None] & key_mask[None, :]
                 & (keys[None, :] >= lo[:, None])
                 & (keys[None, :] < hi[:, None]))
        tl.store(out + rows[:, None] * n + keys[None, :], tl.trans(vals),
                 mask=valid, eviction_policy="evict_first")


def _make_kernel(warp_specialize: bool, reorder: bool):
    """Build a kernel variant; constexpr flags keep one source of truth."""

    @triton.jit
    def _kernel(q, k, ks, w, cu_ks, cu_ke, out, m, n, mt, nt,
                BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
                WS: tl.constexpr, RE: tl.constexpr):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        dims = tl.arange(0, 128)
        heads = tl.arange(0, 64)
        mw = tl.minimum(mt, workers)
        mlane = worker % mw
        nlane = worker // mw
        nw = tl.cdiv(workers, mw)
        for pid_m in range(mlane, mt, mw):
            rows = pid_m * BM + tl.arange(0, BM)
            row_mask = rows < m
            qr = tl.arange(0, BM * 64)
            qrows = pid_m * BM + qr // 64
            qheads = qr % 64
            q_tile = tl.load(
                q + qrows[:, None] * (64 * 128) + qheads[:, None] * 128
                + dims[None, :],
                mask=(qrows < m)[:, None], other=0,
                eviction_policy="evict_last",
            )
            hw = tl.load(w + rows[:, None] * 64 + heads[None, :],
                         mask=row_mask[:, None], other=0.0,
                         eviction_policy="evict_last")
            lo = tl.load(cu_ks + rows, mask=row_mask, other=0)
            hi = tl.load(cu_ke + rows, mask=row_mask, other=0)

            # tl.range/tle.range must appear literally in the for statement:
            # the Triton frontend inspects the AST Call node, so the iterator
            # cannot be hoisted into a variable.
            if RE and WS:
                for pid_n in tle.range(nlane, nt, nw, num_stages=STAGES,
                                       loop_unroll_factor=2, reorder=True,
                                       warp_specialize=True):
                    _n_tile(pid_n, q_tile, hw, lo, hi, row_mask, rows,
                            k, ks, out, n, BM, BN)
            elif RE:
                for pid_n in tle.range(nlane, nt, nw, num_stages=STAGES,
                                       loop_unroll_factor=2, reorder=True):
                    _n_tile(pid_n, q_tile, hw, lo, hi, row_mask, rows,
                            k, ks, out, n, BM, BN)
            elif WS:
                for pid_n in tl.range(nlane, nt, nw, num_stages=STAGES,
                                      warp_specialize=True):
                    _n_tile(pid_n, q_tile, hw, lo, hi, row_mask, rows,
                            k, ks, out, n, BM, BN)
            else:
                for pid_n in tl.range(nlane, nt, nw, num_stages=STAGES):
                    _n_tile(pid_n, q_tile, hw, lo, hi, row_mask, rows,
                            k, ks, out, n, BM, BN)

    return _kernel


def _latency_us(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--window-ratio", type=float, default=0.5)
    ap.add_argument("--block-m", type=int, default=8)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(29)
    m, n = args.m, args.n
    dev = "cuda"
    q = torch.randint(-127, 128, (m, 64, 128), device=dev, dtype=torch.int8)
    k = torch.randint(-127, 128, (n, 128), device=dev, dtype=torch.int8)
    ks = torch.rand(n, device=dev, dtype=torch.float32) / 127.0
    w = torch.randn(m, 64, device=dev, dtype=torch.float32)
    win = int(n * args.window_ratio)
    lo = torch.zeros(m, device=dev, dtype=torch.int32)
    hi = torch.full((m,), win, device=dev, dtype=torch.int32)

    ref = int8_mqa_logits(q, k, ks, w, lo, hi)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    BM, BN = args.block_m, args.block_n
    mt, nt = triton.cdiv(m, BM), triton.cdiv(n, BN)

    print(f"shape M={m} N={n} window={win} BLOCK_M={BM} BLOCK_N={BN} "
          f"stages={args.stages} SM={sm}")
    print(f"{'variant':<10}{'us':>11}{'vs base':>10}  status")

    base_us = None
    for name, ws, re in [("base", False, False), ("ws", True, False),
                         ("reorder", False, True), ("wsre", True, True)]:
        kern = _make_kernel(ws, re)
        out = torch.zeros(m, n, device=dev, dtype=torch.float32)

        def run(kern=kern, out=out, ws=ws, re=re):
            kern[(sm,)](q, k, ks, w, lo, hi, out, m, n, mt, nt,
                        BM=BM, BN=BN, STAGES=args.stages, WS=ws, RE=re,
                        num_warps=8, num_stages=args.stages)

        try:
            run()
            torch.cuda.synchronize()
        except Exception as exc:
            msg = str(exc).strip().splitlines()[-1][:60]
            print(f"{name:<10}{'-':>11}{'-':>10}  COMPILE FAIL: {msg}")
            continue

        ok = torch.equal(out[:, :win], ref[:, :win])
        us = _latency_us(run, repeats=args.repeats)
        if base_us is None:
            base_us = us
        rel = f"{base_us / us:.3f}x"
        print(f"{name:<10}{us:>11.1f}{rel:>10}  "
              f"{'ok' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
