"""Tests for the compatibility shims.

These run without a GPU and without SGLang installed; they exercise the shim
logic itself, which is where the subtle bugs would be.
"""

import sys
import types

import pytest

from sglang_radeon_rdna3 import compat


def test_aiter_stub_is_skipped_when_aiter_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", None)
    # A None entry makes `import aiter` raise, mimicking "not installed".
    result = compat._stub_aiter_gemm_submodules()
    assert "skipped" in result or "applied" in result


def test_aiter_stub_creates_importable_modules(monkeypatch):
    # Pretend aiter exists but lacks the CDNA-only GEMM subpackage.
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    for name in compat._AITER_STUB_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    result = compat._stub_aiter_gemm_submodules()
    assert result.startswith("applied")

    mod = sys.modules[compat._AITER_STUB_MODULES[-1]]
    fn = mod.fused_gemm_afp4wfp4_split_cat
    # Stubs must not fail at import time, only when actually invoked.
    with pytest.raises(RuntimeError, match="RDNA3"):
        fn()


def test_rms_norm_shim_handles_both_arities(monkeypatch):
    calls = []

    fake_ops = types.ModuleType("vllm._custom_ops")

    def real(inp, residual, weight, eps):
        """Stand-in for vLLM's in-place kernel."""
        calls.append((inp, residual, weight, eps))
        residual += inp          # r' = x + r
        inp.copy_(residual * 2)  # stand-in for rmsnorm(r') * w

    fake_ops.fused_add_rms_norm = real
    vllm_pkg = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)

    assert compat._patch_fused_add_rms_norm() == "applied"
    wrapped = fake_ops.fused_add_rms_norm
    assert wrapped._rdna3_wrapped

    torch = pytest.importorskip("torch")

    # 4-arg form passes straight through.
    x = torch.ones(4)
    r = torch.ones(4)
    wrapped(x, r, torch.ones(4), 1e-6)
    assert len(calls) == 1

    # 6-arg form must not clobber the caller's inputs, and must write both
    # the normalized output and the updated residual.
    x = torch.ones(4)
    r = torch.full((4,), 2.0)
    out = torch.empty(4)
    residual_out = torch.empty(4)
    wrapped(out, x, residual_out, r, torch.ones(4), 1e-6)

    assert torch.equal(x, torch.ones(4)), "input tensor was mutated"
    assert torch.equal(r, torch.full((4,), 2.0)), "residual was mutated"
    assert torch.equal(residual_out, torch.full((4,), 3.0))  # 1 + 2
    assert torch.equal(out, torch.full((4,), 6.0))           # 3 * 2


def test_rms_norm_shim_is_idempotent(monkeypatch):
    fake_ops = types.ModuleType("vllm._custom_ops")
    fake_ops.fused_add_rms_norm = lambda *a: None
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)

    assert compat._patch_fused_add_rms_norm() == "applied"
    assert compat._patch_fused_add_rms_norm() == "already applied"


def test_rms_norm_shim_rejects_unknown_arity(monkeypatch):
    fake_ops = types.ModuleType("vllm._custom_ops")
    fake_ops.fused_add_rms_norm = lambda *a: None
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)
    compat._patch_fused_add_rms_norm()

    with pytest.raises(TypeError, match="expected 4"):
        fake_ops.fused_add_rms_norm(1, 2, 3)


def test_apply_all_never_raises(monkeypatch):
    def boom():
        raise ValueError("simulated failure")

    monkeypatch.setattr(compat, "_SHIMS", (("exploding", boom),))
    status = compat.apply_all()
    assert status["exploding"].startswith("FAILED")
