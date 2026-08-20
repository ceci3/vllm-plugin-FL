# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

from vllm_fl.dispatch.backends.vendor.gcu.impl import causal_conv1d


class _FakeKernel:
    marker = "wrapped"

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return grid, args, kwargs

        return launch


def test_block_n_kernel_proxy_overrides_launch_meta():
    proxy = causal_conv1d._BlockNKernelProxy(_FakeKernel(), block_n=2048)

    grid, args, kwargs = proxy["grid"](1, BLOCK_N=256, other=True)

    assert grid == "grid"
    assert args == (1,)
    assert kwargs == {"BLOCK_N": 2048, "other": True}
    assert proxy.marker == "wrapped"


def test_apply_causal_conv1d_gcu_patch_is_idempotent(monkeypatch):
    module = SimpleNamespace(
        _causal_conv1d_fwd_kernel=_FakeKernel(),
        _causal_conv1d_update_kernel=_FakeKernel(),
    )
    monkeypatch.setattr(causal_conv1d.importlib, "import_module", lambda _: module)

    causal_conv1d.apply_causal_conv1d_gcu_patch()
    fwd_proxy = module._causal_conv1d_fwd_kernel
    update_proxy = module._causal_conv1d_update_kernel
    causal_conv1d.apply_causal_conv1d_gcu_patch()

    assert module._causal_conv1d_fwd_kernel is fwd_proxy
    assert module._causal_conv1d_update_kernel is update_proxy
    assert fwd_proxy.block_n == 2048
    assert update_proxy.block_n == 1024
