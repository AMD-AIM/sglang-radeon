"""Compatibility shims that must run before ``sglang`` is imported.

Every entry here works around a specific upstream defect. Each one records
whether it actually fired so ``sglang-rdna3 doctor`` can report the real state
of the installation rather than assuming. As fixes land upstream the
corresponding shim becomes a no-op and can be deleted.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

#: Populated by :func:`apply_all`; maps shim name -> human-readable status.
STATUS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Shim 1 -- stub CDNA-only aiter GEMM submodules.
#
# sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py does:
#     if _is_hip:
#         from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...
# Released aiter builds place those kernels directly under aiter/ops/triton/
# with no gemm/fused subpackage, so the import raises ModuleNotFoundError.
# That happens inside layers/quantization/__init__.py, which sits on the import
# path of *every* model module -- so `import sglang` registers almost nothing.
#
# These are MXFP4 GEMMs for CDNA. RDNA3 has no native FP8/MXFP4 support and
# cannot use them, so stubbing is safe: we only raise if something calls them.
# ---------------------------------------------------------------------------
_AITER_STUB_MODULES = (
    "aiter.ops.triton.gemm",
    "aiter.ops.triton.gemm.fused",
    "aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat",
)


def _stub_aiter_gemm_submodules() -> str:
    try:
        import aiter  # noqa: F401
    except Exception:
        return "skipped (aiter not installed)"

    try:
        import aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat  # noqa: F401
        return "not needed (aiter provides the real modules)"
    except Exception:
        pass

    created = []
    for name in _AITER_STUB_MODULES:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []  # mark as a package so submodules can be added
            sys.modules[name] = module
            created.append(name)

    def _unsupported(*_args: Any, **_kwargs: Any):
        raise RuntimeError(
            "MXFP4/quark fused GEMM is a CDNA-only aiter kernel; it is not "
            "available on RDNA3 (gfx110x). Serve BF16/FP16 weights instead."
        )

    leaf = sys.modules[_AITER_STUB_MODULES[-1]]
    leaf.fused_gemm_afp4wfp4_split_cat = _unsupported
    return f"applied ({len(created)} stub modules)"


# ---------------------------------------------------------------------------
# Shim 2 -- reconcile the fused_add_rms_norm signature.
#
# sglang/srt/layers/layernorm.py calls the legacy 6-arg out-of-place form
#     fused_add_rms_norm(out, input, residual_out, residual, weight, eps)
# but vLLM >= 0.16 ships a 4-arg in-place kernel
#     fused_add_rms_norm(input, residual, weight, epsilon) -> None
# The first forward pass of any Gemma-style RMSNorm model then dies with
#     TypeError: takes 4 positional arguments but 6 were given
# Qwen3.8-27B hits this immediately. Note this is not RDNA3-specific -- it is
# sglang tracking a changed vLLM internal API -- but it blocks us regardless.
#
# The in-place kernel computes, for tensors x and r:
#     r' = x + r            (written back into the residual argument)
#     out = rmsnorm(r') * w (written back into the input argument)
# so the 6-arg form is emulated by staging copies in the caller's buffers.
# ---------------------------------------------------------------------------
def _patch_fused_add_rms_norm() -> str:
    try:
        import vllm._custom_ops as ops
    except Exception:
        return "skipped (vllm not installed)"

    real = getattr(ops, "fused_add_rms_norm", None)
    if real is None:
        return "skipped (vllm has no fused_add_rms_norm)"
    if getattr(real, "_rdna3_wrapped", False):
        return "already applied"

    def fused_add_rms_norm(*args: Any):
        if len(args) == 4:
            return real(*args)
        if len(args) == 6:
            out, x, residual_out, residual, weight, eps = args
            out.copy_(x)
            residual_out.copy_(residual)
            real(out, residual_out, weight, eps)
            return None
        raise TypeError(
            f"fused_add_rms_norm() got {len(args)} positional arguments; "
            "expected 4 (in-place) or 6 (legacy out-of-place)"
        )

    fused_add_rms_norm._rdna3_wrapped = True  # type: ignore[attr-defined]
    ops.fused_add_rms_norm = fused_add_rms_norm
    return "applied"


_SHIMS = (
    ("aiter_gemm_stubs", _stub_aiter_gemm_submodules),
    ("fused_add_rms_norm_arity", _patch_fused_add_rms_norm),
)


def apply_all() -> dict[str, str]:
    """Apply every compatibility shim. Safe to call more than once."""
    for name, fn in _SHIMS:
        try:
            STATUS[name] = fn()
        except Exception as exc:  # never block startup on a shim
            STATUS[name] = f"FAILED: {exc}"
            logger.warning("RDNA3 compat shim %r failed: %s", name, exc)
    return STATUS
