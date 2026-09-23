"""Tests for the compatibility shims.

These run without a GPU and without SGLang installed; they exercise the shim
logic itself, which is where the subtle bugs would be.
"""

import sys
import types

import pytest

from sglang_radeon_rdna3 import compat


@pytest.fixture
def absent_aiter(monkeypatch):
    """Run the shim in its "aiter is not installed at all" mode.

    Yields the status string. Teardown matters here: the shim installs a
    sys.meta_path finder, which is process-global, so leaving it behind would
    hand every later test a fabricated `aiter`.
    """
    for name in [n for n in sys.modules if n == "aiter" or n.startswith("aiter.")]:
        monkeypatch.delitem(sys.modules, name)
    # The shim sets this once and leaves it; clear it so each test sees a
    # first run rather than the previous test's leftovers.
    monkeypatch.delenv("USE_ROCM_AITER_ROPE_BACKEND", raising=False)
    # A None entry is what an absent module looks like to `import`.
    monkeypatch.setitem(sys.modules, "aiter", None)
    status = compat._stub_aiter_gemm_submodules()
    monkeypatch.delitem(sys.modules, "aiter")  # the finder answers from here on
    try:
        yield status
    finally:
        sys.meta_path[:] = [
            f for f in sys.meta_path
            if not isinstance(f, compat._AiterPlaceholderFinder)
        ]
        for name in [n for n in list(sys.modules)
                     if n == "aiter" or n.startswith("aiter.")]:
            if getattr(sys.modules[name], "_rdna3_placeholder", False):
                del sys.modules[name]


def test_absent_aiter_gets_a_whole_package_placeholder(absent_aiter):
    """The regression that shipped a green image: aiter missing entirely.

    The older shim returned "skipped (aiter not installed)" here and left
    sglang to die at import. Anything but a placeholder is that bug again.
    """
    assert absent_aiter.startswith("applied"), absent_aiter

    import aiter

    assert aiter._rdna3_placeholder is True

    # Arbitrary depth, because the set of modules sglang reaches for moves
    # with every upstream bump -- enumerating them is what failed before.
    from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: F401
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (  # noqa: F401,E501
        fused_gemm_afp4wfp4_split_cat,
    )
    import aiter.some.module.nobody.has.written.yet  # noqa: F401


def test_placeholder_refuses_to_run(absent_aiter):
    """Inert on import, loud on call. A silent no-op would corrupt output."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    with pytest.raises(RuntimeError, match="CDNA-only"):
        dynamic_mxfp4_quant()
    # The message has to name the attribute, or a mid-pass failure says
    # nothing about which kernel was reached.
    with pytest.raises(RuntimeError, match=r"aiter\.ops\.triton\.quant"):
        dynamic_mxfp4_quant()


def test_placeholder_leaves_dunders_to_python(absent_aiter):
    """importlib, inspect and pickle probe for dunders and expect to fail."""
    import aiter

    with pytest.raises(AttributeError):
        aiter.__wrapped__


def test_placeholder_disarms_apex_rope_probe(absent_aiter, monkeypatch):
    """apex picks its RoPE kernel by `try: import aiter`.

    A placeholder that satisfies that probe moves the failure from import time
    into the middle of a forward pass, which is worse than the crash we set
    out to fix.
    """
    import os

    assert os.environ["USE_ROCM_AITER_ROPE_BACKEND"] == "0"
    assert "apex RoPE" in absent_aiter


def test_placeholder_does_not_override_an_explicit_rope_choice(monkeypatch):
    """Someone who set the var has a reason; the shim is not the authority."""
    monkeypatch.setenv("USE_ROCM_AITER_ROPE_BACKEND", "1")
    assert compat._disarm_aiter_presence_probes() == ""
    import os

    assert os.environ["USE_ROCM_AITER_ROPE_BACKEND"] == "1"


def test_placeholder_install_is_idempotent(absent_aiter):
    before = sum(
        isinstance(f, compat._AiterPlaceholderFinder) for f in sys.meta_path
    )
    assert compat._install_aiter_placeholder().startswith("already applied")
    after = sum(
        isinstance(f, compat._AiterPlaceholderFinder) for f in sys.meta_path
    )
    assert before == after == 1


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

    monkeypatch.setattr(compat, "_SHIMS", (("exploding", boom, False),))
    status = compat.apply_all()
    assert status["exploding"].startswith("FAILED")


def test_deferrable_shim_reports_deferred_not_skipped(monkeypatch):
    """A shim whose target is not importable *yet* must say so.

    Reporting "skipped" is indistinguishable from "not applicable", which is
    how nested_weight_detection went a whole release never applying while
    doctor -- run later, with everything imported -- called it healthy.
    """
    monkeypatch.setattr(
        compat, "STATUS", {}, raising=False
    )
    monkeypatch.setattr(
        compat,
        "_SHIMS",
        (
            ("waits", lambda: "skipped (sglang-diffusion not available)", True),
            ("final", lambda: "skipped (vllm not installed)", False),
        ),
    )
    status = compat.apply_all()
    assert status["waits"] == "deferred (sglang-diffusion not available)"
    assert status["final"] == "skipped (vllm not installed)"
    assert compat.deferred_shims() == ["waits"]


def test_apply_all_does_not_rerun_applied_shims(monkeypatch):
    """Some shims walk whole package trees; the bootstrap may retry often."""
    calls = []
    monkeypatch.setattr(compat, "STATUS", {}, raising=False)
    monkeypatch.setattr(
        compat,
        "_SHIMS",
        (("counts", lambda: calls.append(1) or "applied (1 thing)", True),),
    )
    compat.apply_all()
    compat.apply_all()
    assert len(calls) == 1
    assert compat.deferred_shims() == []


def _fake_lora_module(monkeypatch):
    """Stand in for sglang's LoRA linear module, shaped like the real one.

    The real ``BaseLayerWithLoRA`` delegates ``weight`` and ``bias`` to the
    wrapped layer and stops there; ``quant_method`` is reachable only as
    ``self.base_layer.quant_method``. Plain classes raise AttributeError for a
    missing attribute exactly like ``nn.Module.__getattr__`` does, so no torch
    is needed to reproduce the failure.
    """

    class BaseLayerWithLoRA:
        def __init__(self, base_layer):
            self.base_layer = base_layer

        @property
        def weight(self):
            return self.base_layer.weight

    class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
        pass

    mod = types.ModuleType("sglang.multimodal_gen.runtime.layers.lora.linear")
    mod.BaseLayerWithLoRA = BaseLayerWithLoRA
    mod.RowParallelLinearWithLoRA = RowParallelLinearWithLoRA

    pkg = "sglang.multimodal_gen.runtime.layers.lora.linear"
    parts = pkg.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, pkg, mod)
    return mod


def test_lora_wrapper_gets_quant_method_from_its_base_layer(monkeypatch):
    mod = _fake_lora_module(monkeypatch)

    class Base:
        weight = "w"
        quant_method = "the real quant method"

    layer = mod.RowParallelLinearWithLoRA(Base())

    # This is the bug: H3's _accepts_mxfp8_input reads it with no default.
    with pytest.raises(AttributeError, match="quant_method"):
        layer.quant_method

    assert compat._patch_lora_quant_method_delegation() == "applied"

    # Delegated, not stubbed -- on hardware that does have MXFP8 the predicate
    # must still get the base layer's real answer.
    assert layer.quant_method == "the real quant method"
    assert compat._patch_lora_quant_method_delegation() == "already applied"


def test_lora_quant_method_is_none_when_the_base_layer_has_none(monkeypatch):
    mod = _fake_lora_module(monkeypatch)

    class Base:
        weight = "w"

    layer = mod.BaseLayerWithLoRA(Base())
    assert compat._patch_lora_quant_method_delegation() == "applied"
    # _accepts_mxfp8_input short-circuits on None; it must not raise.
    assert layer.quant_method is None


def test_lora_quant_method_shim_stands_down_once_upstream_fixes_it(monkeypatch):
    mod = _fake_lora_module(monkeypatch)
    mod.BaseLayerWithLoRA.quant_method = property(
        lambda self: self.base_layer.quant_method
    )
    assert compat._patch_lora_quant_method_delegation().startswith("not needed")


def test_lora_quant_method_shim_defers_when_sglang_is_absent(monkeypatch):
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", [])
    status = compat._patch_lora_quant_method_delegation()
    assert status.startswith("skipped")
    # Deferrable, so apply_all turns that "skipped" into a retry rather than
    # recording it as a permanent pass.
    assert dict((n, d) for n, _, d in compat._SHIMS)["lora_quant_method"] is True
