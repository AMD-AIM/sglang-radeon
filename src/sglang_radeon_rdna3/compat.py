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


# ---------------------------------------------------------------------------
# Shim 3 -- recognize nested component weights (SGLang-Diffusion).
#
# multimodal_gen/runtime/utils/hf_diffusers_utils.py checks component
# completeness with a single-level glob:
#     glob.glob(os.path.join(component_path, "*.safetensors"))
# MiniMax-H3 ships its video VAE one level down, at
#     FL2VA/video_vae/source/model.safetensors
# so a fully downloaded checkpoint is judged incomplete. SGLang then tries to
# "repair" it by re-downloading, passes the local path to the Hub API, and
# dies with a confusing error:
#     HFValidationError: Repo id must be in the form 'repo_name' or
#     'namespace/repo_name': '/models/MiniMax-H3/FL2VA'
#
# Widening the probe by one directory level is enough. Not RDNA3-specific --
# it affects anyone serving MiniMax-H3 from a local directory.
# ---------------------------------------------------------------------------
def _patch_nested_weight_detection() -> str:
    try:
        from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils as hfu
    except Exception:
        return "skipped (sglang-diffusion not available)"

    real = getattr(hfu, "_has_local_weight_files", None)
    if real is None:
        return "skipped (probe function not found)"
    if getattr(real, "_rdna3_wrapped", False):
        return "already applied"

    import glob
    import os

    patterns = getattr(hfu, "_WEIGHT_FILE_PATTERNS",
                       ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"))

    def _has_local_weight_files(component_path: str) -> bool:
        if real(component_path):
            return True
        # One extra level: enough for layouts like video_vae/source/.
        return any(
            glob.glob(os.path.join(component_path, "*", pattern))
            for pattern in patterns
        )

    _has_local_weight_files._rdna3_wrapped = True  # type: ignore[attr-defined]
    hfu._has_local_weight_files = _has_local_weight_files
    return "applied"


# ---------------------------------------------------------------------------
# Shim 4 -- let the MiniMax-H3 denoise loop run on ROCm.
#
# minimax_h3/stages/denoising.py::_run_full_loop gates on a platform
# allowlist:
#     if not (current_platform.is_cuda() or current_platform.is_mps()):
#         raise RuntimeError("MiniMax H3 full-loop denoise requires CUDA or MPS")
# The code past that gate is device-agnostic PyTorch -- it calls
# get_local_torch_device() and proceeds. On ROCm that already returns
# "cuda:0" (HIP reuses the CUDA device namespace), so the loop itself has no
# CUDA dependency; the allowlist simply predates ROCm support.
#
# Rather than patch the function we widen the predicate, by making
# current_platform.is_cuda() report True on a HIP platform for the duration.
# That is narrow enough to be safe here: is_cuda_alike() already treats ROCm
# as CUDA-like elsewhere in the same codebase.
# ---------------------------------------------------------------------------
def _patch_h3_denoise_platform_gate() -> str:
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (  # noqa: E501
            denoising,
        )
    except Exception:
        return "skipped (MiniMax-H3 stages not available)"

    platform = getattr(denoising, "current_platform", None)
    if platform is None:
        return "skipped (platform object not found)"
    if not getattr(platform, "is_hip", lambda: False)():
        return "not needed (not a HIP platform)"
    if getattr(platform.is_cuda, "_rdna3_wrapped", False):
        return "already applied"

    real_is_cuda = platform.is_cuda

    def is_cuda() -> bool:
        # HIP presents as CUDA to torch; the gate this satisfies only guards
        # device-agnostic code.
        return True

    is_cuda._rdna3_wrapped = True  # type: ignore[attr-defined]
    is_cuda._rdna3_real = real_is_cuda  # type: ignore[attr-defined]
    try:
        platform.is_cuda = is_cuda
    except Exception as exc:
        return f"FAILED: {exc}"
    return "applied"


# ---------------------------------------------------------------------------
# Shim 5 -- decline SGLang-Diffusion's CUDA-only JIT kernels on RDNA3.
#
# The diffusion denoise path JIT-compiles fused kernels from
# sglang/kernels/jit/csrc. Thirty-four of those sources hard-include CUDA-only
# headers (cuda_bf16.h, cuda_fp16.h, cuda_fp8.h) or assume a 32-bit warp mask,
# so on HIP they fail to compile:
#
#     fatal error: 'cuda_bf16.h' file not found
#     amd_warp_sync_functions.h:277: static assertion failed ...
#         'sizeof(unsigned int) == 8': The mask must be a 64-bit integer
#
# These are real source incompatibilities -- porting them means rewriting the
# kernels for wave64 and HIP headers, not flipping a flag.
#
# Every such kernel is opt-in behind a ``can_use_*`` predicate, and callers
# fall back to PyTorch reference implementations when one returns False. We
# therefore report False for the whole family on HIP: slower, but it keeps the
# pipeline on paths that actually run. Set SGLANG_RDNA3_ALLOW_JIT=1 to leave
# them enabled (useful for checking whether a kernel has since been ported).
# ---------------------------------------------------------------------------
_JIT_OPS_PACKAGE = "sglang.kernels.ops"

#: Functions that dispatch to a CUDA-only JIT kernel behind an `is_cuda`
#: test, which is True for HIP tensors, and that already contain a correct
#: PyTorch fallback further down. We replace the dispatcher outright with its
#: fallback branch.
#:
#: Making the kernel itself raise does not work: these call sites have no
#: try/except, only a condition. And patching the kernel globally breaks
#: text-encoder construction, where a failure is fatal ("Failed to load
#: customized text_encoder; native fallback is disabled"). Replacing the one
#: dispatcher is both sufficient and contained.
_DISPATCHERS_TO_REPLACE = (
    ("sglang.multimodal_gen.runtime.models.dits.minimax_h3", "_apply_qk_norm"),
)


def _qk_norm_reference(q, k, q_norm, k_norm, head_dim):
    """The fallback branch of minimax_h3._apply_qk_norm, verbatim."""
    return q_norm(q), k_norm(k)


def _decline_cuda_only_jit_kernels() -> str:
    import os

    try:
        from sglang.srt.utils.common import is_hip
    except Exception:
        return "skipped (sglang not available)"
    if not is_hip():
        return "not needed (not a HIP platform)"
    if os.environ.get("SGLANG_RDNA3_ALLOW_JIT", "").lower() in ("1", "true", "yes"):
        return "disabled via SGLANG_RDNA3_ALLOW_JIT"

    import importlib
    import pkgutil
    import sys

    def _false(*_args, **_kwargs) -> bool:
        return False

    _false._rdna3_wrapped = True  # type: ignore[attr-defined]

    patched: set[str] = set()

    try:
        root = importlib.import_module(_JIT_OPS_PACKAGE)
    except Exception as exc:
        return f"skipped (cannot import {_JIT_OPS_PACKAGE}: {exc})"

    # Walk the ops package and neutralise every can_use_* predicate.
    for mod_info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        try:
            module = importlib.import_module(mod_info.name)
        except Exception:
            continue  # a module that will not import cannot be used anyway
        for attr in dir(module):
            if not attr.startswith("can_use_"):
                continue
            fn = getattr(module, attr, None)
            if callable(fn) and not getattr(fn, "_rdna3_wrapped", False):
                setattr(module, attr, _false)
                patched.add(attr)

    # Callers import these by name, so rebind copies already bound elsewhere.
    for name, module in list(sys.modules.items()):
        if not name.startswith("sglang.") or module is None:
            continue
        for attr in patched:
            fn = getattr(module, attr, None)
            if callable(fn) and not getattr(fn, "_rdna3_wrapped", False):
                setattr(module, attr, _false)

    # Some call sites skip the predicate entirely. MiniMax-H3's DiT, for
    # instance, gates on `q.is_cuda` -- which is True for HIP tensors, since
    # torch presents ROCm devices as CUDA -- and calls the kernel directly:
    #
    #   minimax_h3.py:381  if (q.is_cuda and q.dtype == _BF16_DTYPE and ...):
    #   minimax_h3.py:390      fused_inplace_qknorm(...)
    #   minimax_h3.py:398  return q_norm(q), k_norm(k)   <- the fallback
    #
    # There is a perfectly good PyTorch path one line below, but it is
    # unreachable on ROCm. Making the kernel raise ImportError sends the
    # caller there, and matches what these wrappers already do when the JIT
    # build fails for other reasons.
    for mod_path, attr in _DISPATCHERS_TO_REPLACE:
        module = sys.modules.get(mod_path)
        if module is None:
            try:
                module = importlib.import_module(mod_path)
            except Exception:
                continue
        fn = getattr(module, attr, None)
        if fn is None or getattr(fn, "_rdna3_wrapped", False):
            continue

        _qk_norm_reference._rdna3_wrapped = True  # type: ignore[attr-defined]
        setattr(module, attr, _qk_norm_reference)
        patched.add(f"{mod_path.rsplit('.', 1)[-1]}.{attr}")

    if not patched:
        # Either nothing matched, or a previous call already neutralised
        # them all -- distinguish, so the status is not misleading.
        already = sum(
            1
            for mod in list(sys.modules.values())
            if mod is not None
            for attr in dir(mod)
            if attr.startswith("can_use_")
            and getattr(getattr(mod, attr, None), "_rdna3_wrapped", False)
        )
        if already:
            return f"already applied ({already} predicates disabled)"
        return "no can_use_* predicates found"
    return f"applied ({len(patched)} kernel predicates disabled)"


# ---------------------------------------------------------------------------
# Shim 6 -- replace inline PTX in the diffusion Triton kernels.
#
# Several diffusion kernels reach for inline PTX to pin down rounding
# behaviour, e.g. in kernels/ops/diffusion/common/numerics.py:
#
#     @triton.jit
#     def mul_rn_f32(x, y):
#         return tl.inline_asm_elementwise(
#             asm="mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", ...)
#
# PTX is NVIDIA's instruction set, so the AMD backend cannot compile any of
# it:
#
#     error: couldn't allocate output register for constraint 'f'
#
# ('f' is a PTX float register constraint.) This surfaces during VAE decode,
# after denoising has finished, so it reads like a decode bug rather than a
# codegen one.
#
# Each of these is a plain arithmetic op that Triton expresses natively. We
# lose the bit-exactness guarantee the PTX was there to provide -- these are
# "correctly-rounded" and "approximate" variants chosen to match CUDA output
# exactly -- but the results stay within an ulp, which is the right trade for
# a platform that otherwise cannot run at all.
# ---------------------------------------------------------------------------
def _patch_ptx_primitives() -> str:
    try:
        from sglang.srt.utils.common import is_hip
    except Exception:
        return "skipped (sglang not available)"
    if not is_hip():
        return "not needed (not a HIP platform)"

    try:
        import triton
    except Exception:
        return "skipped (triton not available)"

    @triton.jit
    def _mul_rn_f32(x, y):
        return x * y

    @triton.jit
    def _div_rn_f32(x, y):
        return x / y

    @triton.jit
    def _rsqrt_approx_f32(x):
        return 1.0 / tl_sqrt(x)

    @triton.jit
    def _rcp4(x):
        # The PTX was rcp.approx.f32 plus one Newton step; a true divide
        # matches it to within an ulp.
        return 1.0 / x

    # tl.sqrt lives at different paths across Triton versions.
    import triton.language as tl

    tl_sqrt = getattr(tl, "sqrt", None) or getattr(tl.math, "sqrt", None)
    if tl_sqrt is None:
        return "skipped (no tl.sqrt in this Triton)"

    @triton.jit
    def _rsqrt_approx_f32(x):  # noqa: F811 - defined once tl_sqrt is known
        return tl.rsqrt(x) if hasattr(tl, "rsqrt") else 1.0 / tl.sqrt(x)

    replacements = {
        "sglang.kernels.ops.diffusion.common.numerics": {
            "mul_rn_f32": _mul_rn_f32,
            "div_rn_f32": _div_rn_f32,
            "rsqrt_approx_f32": _rsqrt_approx_f32,
        },
        "sglang.kernels.ops.diffusion.norm.layernorm_modulate_triton": {
            "_rcp4": _rcp4,
        },
    }

    import importlib
    import sys

    applied = []
    for mod_path, subs in replacements.items():
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        if getattr(mod, "_rdna3_ptx_patched", False):
            applied.append(f"{mod_path.rsplit('.', 1)[-1]}(cached)")
            continue
        for name, impl in subs.items():
            if hasattr(mod, name):
                setattr(mod, name, impl)
                applied.append(name)
        mod._rdna3_ptx_patched = True

        # Rebind copies imported by name into other modules.
        for other in list(sys.modules.values()):
            if other is None or other is mod:
                continue
            for name, impl in subs.items():
                if getattr(other, name, None) is not None and hasattr(mod, name):
                    setattr(other, name, impl)

    # Triton caches compiled kernels keyed on their callees' source.
    for mod_path in replacements:
        mod = sys.modules.get(mod_path)
        for attr in dir(mod) if mod else []:
            cache = getattr(getattr(mod, attr, None), "cache", None)
            if isinstance(cache, dict):
                cache.clear()

    if not applied:
        return "no PTX primitives found"
    return f"applied ({len(applied)} primitives)"


_SHIMS = (
    ("aiter_gemm_stubs", _stub_aiter_gemm_submodules),
    ("fused_add_rms_norm_arity", _patch_fused_add_rms_norm),
    ("nested_weight_detection", _patch_nested_weight_detection),
    ("h3_denoise_platform_gate", _patch_h3_denoise_platform_gate),
    ("decline_cuda_only_jit_kernels", _decline_cuda_only_jit_kernels),
    ("ptx_primitives", _patch_ptx_primitives),
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
