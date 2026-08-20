"""Where does the DSV4 INT8 MQA kernel spend its time: dot or epilogue?

The shipped kernel reaches only ~59% of the H20 INT8 peak while
``cutlass_scaled_mm`` reaches 88% on the same GPU, so something other than the
Tensor Core issue rate is costing time. This isolates the epilogue by keeping
the loads, window test and ``tl.dot`` identical and swapping only the reduction:

  full : production epilogue (fp32 scale, relu, weighted sum over 64 heads)
  sum  : plain int32 sum over heads (no scale/relu/weight)
  none : no cross-head reduction at all (slice head 0)

Only ``full`` is numerically useful; the other two exist to price the epilogue.
"""

import argparse

import torch
import triton.experimental.tle.language as tle
from vllm.triton_utils import tl, triton


@triton.jit
def _kernel(q, k, ks, w, cu_ks, cu_ke, out, m, n, mt, nt,
            BM: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr,
            MODE: tl.constexpr):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    dims = tl.arange(0, 128)
    heads = tl.arange(0, 64)
    mw = tl.minimum(mt, workers)
    ml = worker % mw
    nl = worker // mw
    nw = tl.cdiv(workers, mw)
    for pid_m in range(ml, mt, mw):
        rows = pid_m * BM + tl.arange(0, BM)
        rmask = rows < m
        qr = tl.arange(0, BM * 64)
        qrows = pid_m * BM + qr // 64
        qh = qr % 64
        qt = tl.load(q + qrows[:, None] * (64 * 128) + qh[:, None] * 128
                     + dims[None, :],
                     mask=(qrows < m)[:, None], other=0,
                     eviction_policy="evict_last")
        hw = tl.load(w + rows[:, None] * 64 + heads[None, :],
                     mask=rmask[:, None], other=0.0,
                     eviction_policy="evict_last")
        lo = tl.load(cu_ks + rows, mask=rmask, other=0)
        hi = tl.load(cu_ke + rows, mask=rmask, other=0)
        lmin = tl.min(tl.where(rmask, lo, n), axis=0)
        hmax = tl.max(tl.where(rmask, hi, 0), axis=0)
        for pid_n in tl.range(nl, nt, nw, num_stages=ST):
            keys = pid_n * BN + tl.arange(0, BN)
            kmask = keys < n
            tmin = pid_n * BN
            tmax = tl.minimum(tmin + BN, n) - 1
            if (tmax >= lmin) & (tmin < hmax):
                kt = tle.load(k + keys[:, None] * 128 + dims[None, :],
                              mask=kmask[:, None], other=0, is_async=True)
                sc = tle.load(ks + keys, mask=kmask, other=0.0, is_async=True)
                dots = tl.dot(kt, tl.trans(qt), out_dtype=tl.int32)
                if MODE == 0:
                    s = tl.reshape(dots, (BN, BM, 64)).to(tl.float32)
                    v = tl.sum(tl.maximum(s * sc[:, None, None], 0.0)
                               * hw[None, :, :], axis=2)
                elif MODE == 1:
                    v = tl.sum(tl.reshape(dots, (BN, BM, 64)), axis=2).to(
                        tl.float32)
                else:
                    v = tl.max(tl.reshape(dots, (BN, BM, 64)), axis=2).to(
                        tl.float32)
                valid = (rmask[:, None] & kmask[None, :]
                         & (keys[None, :] >= lo[:, None])
                         & (keys[None, :] < hi[:, None]))
                tl.store(out + rows[:, None] * n + keys[None, :], tl.trans(v),
                         mask=valid, eviction_policy="evict_first")


def _lat(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--block-m", type=int, default=8)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--peak-tops", type=float, default=296.0)
    args = ap.parse_args()

    m, n, BM, BN, ST = args.m, args.n, args.block_m, args.block_n, args.stages
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(29)
    q = torch.randint(-127, 128, (m, 64, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (n, 128), device="cuda", dtype=torch.int8)
    ks = torch.rand(n, device="cuda") / 127.0
    w = torch.randn(m, 64, device="cuda")
    lo = torch.zeros(m, device="cuda", dtype=torch.int32)
    hi = torch.full((m,), n // 2, device="cuda", dtype=torch.int32)
    mt, nt = triton.cdiv(m, BM), triton.cdiv(n, BN)
    tiles = mt * (nt // 2)
    ops = tiles * BN * (BM * 64) * 128 * 2

    print(f"M={m} N={n} window=50% BLOCK_M={BM} BLOCK_N={BN} stages={ST} "
          f"SM={sm}")
    print(f"{'epilogue':<26}{'us':>9}{'TOPS':>8}{'%peak':>7}{'smem':>8}")
    base = None
    for mode, label in [(0, "full (production)"), (1, "int32 sum only"),
                        (2, "int32 max only")]:
        out = torch.zeros(m, n, device="cuda")

        def run(mode=mode, out=out):
            return _kernel[(sm,)](q, k, ks, w, lo, hi, out, m, n, mt, nt,
                                  BM=BM, BN=BN, ST=ST, MODE=mode,
                                  num_warps=8, num_stages=ST)

        cc = run()
        torch.cuda.synchronize()
        us = _lat(run)
        tops = ops / (us * 1e-6) / 1e12
        if base is None:
            base = us
        print(f"{label:<26}{us:>9.0f}{tops:>8.1f}"
              f"{tops / args.peak_tops * 100:>6.0f}%{cc.metadata.shared:>8}")
    print(f"\nepilogue 占比上限 = {(base - us) / base * 100:.0f}% "
          f"(full 与 no-reduction 之差)")


if __name__ == "__main__":
    main()
