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
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any

logger = logging.getLogger(__name__)

#: Populated by :func:`apply_all`; maps shim name -> human-readable status.
STATUS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Shim 1 -- stand in for the CDNA-only aiter kernels.
#
# sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py does:
#     if _is_hip:
#         from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...
# That happens inside layers/quantization/__init__.py, which sits on the import
# path of *every* model module -- so `import sglang` registers almost nothing.
#
# There are two distinct ways this breaks, and this shim handles both:
#
#   a) aiter is installed but the layout differs. Released aiter builds place
#      those kernels directly under aiter/ops/triton/ with no gemm/fused
#      subpackage, so the import raises ModuleNotFoundError.
#
#   b) aiter is not installed at all -- which is this image's situation, since
#      aiter is a CDNA project we deliberately do not build. sglang treats
#      "running on HIP" as "aiter is present", so `import aiter` itself is the
#      failure:
#          quark/schemes/quark_w4a4_mxfp4.py:86
#              ModuleNotFoundError: No module named 'aiter'
#      SGLANG_USE_AITER=0 does not help: these imports are at module scope and
#      run before any flag is consulted.
#
# Case (b) is not a fixed list of modules -- the sglang tree has 8 top-level
# and 122 conditional aiter imports, and the set moves with every upstream
# bump. Patching them one at a time does not converge, so we fabricate the
# whole `aiter` namespace lazily instead: any `aiter.*` module imports, any
# attribute resolves, and only *calling* one raises.
#
# These are MXFP4/FP8 kernels for CDNA. RDNA3 has no native FP8/MXFP4 support
# and cannot use them, so a placeholder that raises on call is exactly right:
# nothing on a working RDNA3 path ever reaches one.
# ---------------------------------------------------------------------------
_AITER_STUB_MODULES = (
    "aiter.ops.triton.gemm",
    "aiter.ops.triton.gemm.fused",
    "aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat",
)

_AITER_ABSENT_MSG = (
    "aiter is a CDNA-only (gfx942) kernel library and is not installed in this "
    "RDNA3 image; {0} is a placeholder that exists only so `import sglang` "
    "succeeds. Reaching it means a CDNA-only code path was taken on gfx110x -- "
    "serve BF16/FP16 weights instead of MXFP4/FP8 quantized ones."
)


class _AiterPlaceholder(types.ModuleType):
    """A module whose every attribute is a function that refuses to run."""

    def __getattr__(self, name: str) -> Any:
        # Dunders must still fail normally: the import machinery, inspect,
        # and pickle all probe for them and expect AttributeError.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _unavailable(*_args: Any, **_kwargs: Any):
            raise RuntimeError(_AITER_ABSENT_MSG.format(f"{self.__name__}.{name}"))

        _unavailable.__name__ = name
        _unavailable.__qualname__ = f"{self.__name__}.{name}"
        _unavailable._rdna3_placeholder = True  # type: ignore[attr-defined]
        return _unavailable


class _AiterPlaceholderFinder(MetaPathFinder, Loader):
    """Fabricates `aiter` and every submodule under it, on demand."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aiter" or fullname.startswith("aiter."):
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _AiterPlaceholder(spec.name)
        module.__path__ = []  # a package, so submodules resolve through us too
        module._rdna3_placeholder = True
        return module

    def exec_module(self, module):
        pass  # nothing to execute; attributes are synthesized on access


def _disarm_aiter_presence_probes() -> str:
    """Stop libraries that detect aiter *by importing it* from taking the bait.

    Fabricating a module has a cost: code that asks "is aiter here?" by trying
    to import it now gets yes. sglang itself is safe -- it gates every *use* of
    aiter on the SGLANG_USE_AITER env var (off by default), and only its
    module-scope imports were unconditional. apex is not so careful.
    apex/transformer/functional/fused_rope.py does:

        if IS_ROCM_PYTORCH and USE_ROCM_AITER_ROPE_BACKEND:
            try:
                import aiter
                AITER_ROPE_BACKEND = True

    With a placeholder in place that switches RoPE onto a kernel that does not
    exist, and the failure lands in the middle of a forward pass instead of at
    import -- strictly worse than the crash we set out to fix. apex documents
    an off switch; use it, unless the operator has expressed a preference.

    This has to run before apex is imported. The .pth bootstrap guarantees
    that: it fires on the first ``sglang*`` import, and apex only ever gets
    pulled in from inside sglang.
    """
    import os

    if "USE_ROCM_AITER_ROPE_BACKEND" in os.environ:
        return ""  # operator set it explicitly; not ours to override
    os.environ["USE_ROCM_AITER_ROPE_BACKEND"] = "0"
    if "apex.transformer.functional.fused_rope" in sys.modules:
        return "; WARNING: apex was already imported, its RoPE backend may be aiter"
    return "; apex RoPE pinned to its native kernel"


def _install_aiter_placeholder() -> str:
    if any(isinstance(f, _AiterPlaceholderFinder) for f in sys.meta_path):
        return "already applied (aiter placeholder installed)"
    note = _disarm_aiter_presence_probes()
    # Ahead of the real finders, so nothing can half-import a partial aiter.
    sys.meta_path.insert(0, _AiterPlaceholderFinder())
    return f"applied (aiter absent; whole-package placeholder installed{note})"


def _stub_aiter_gemm_submodules() -> str:
    try:
        import aiter  # noqa: F401
    except Exception:
        return _install_aiter_placeholder()

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

        # Capture the originals before overwriting them: they are what
        # identifies a module that did `from numerics import mul_rn_f32`.
        originals = {name: getattr(mod, name, None) for name in subs}

        for name, impl in subs.items():
            if hasattr(mod, name):
                setattr(mod, name, impl)
                applied.append(name)
        mod._rdna3_ptx_patched = True

        # Rebind copies imported by name into other modules.
        #
        # Two things here are deliberate and must not be "simplified" back:
        #
        # 1. Read from __dict__ rather than getattr(). sys.modules holds
        #    objects with a dynamic __getattr__, and torch.ops is one of
        #    them: asking it for any unknown name *fabricates an operator
        #    namespace* and records it. Probing for "mul_rn_f32" therefore
        #    invented torch.ops.mul_rn_f32, the setattr below replaced it
        #    with a Triton function, and torch later walked its own
        #    namespace list and died with
        #        TypeError: attribute name must be string, not 'function'
        #    deep inside inductor's decomposition table -- which surfaced
        #    as `sglang serve` refusing to start at all.
        # 2. Match on identity, not on the name. A module that merely
        #    happens to define something called "_rcp4" is not our copy,
        #    and overwriting it would be a silent corruption of unrelated
        #    code.
        for other in list(sys.modules.values()):
            if other is None or other is mod:
                continue
            namespace = getattr(other, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for name, impl in subs.items():
                if originals[name] is not None and namespace.get(name) is originals[name]:
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


#: ``(name, function, deferrable)``.
#:
#: *deferrable* marks a shim whose target lives inside SGLang itself. Those
#: cannot run on the first pass -- the bootstrap fires mid-``import sglang``,
#: so the submodule they patch is not importable yet -- and a "skipped" from
#: one of them means "ask again later", not "not applicable". They are the
#: shims :mod:`~sglang_radeon_rdna3._bootstrap` retries.
#:
#: ``fused_add_rms_norm_arity`` is deliberately not deferrable: it patches
#: vLLM, so "skipped (vllm not installed)" is a final answer.
_SHIMS = (
    ("aiter_gemm_stubs", _stub_aiter_gemm_submodules, False),
    ("fused_add_rms_norm_arity", _patch_fused_add_rms_norm, False),
    ("nested_weight_detection", _patch_nested_weight_detection, True),
    ("h3_denoise_platform_gate", _patch_h3_denoise_platform_gate, True),
    ("decline_cuda_only_jit_kernels", _decline_cuda_only_jit_kernels, True),
    ("ptx_primitives", _patch_ptx_primitives, True),
)

#: Statuses that mean "this shim is finished, do not run it again".
_SETTLED = ("applied", "already applied")


def deferred_shims() -> list[str]:
    """Shims still waiting for the module they patch to become importable."""
    return [name for name, status in STATUS.items() if status.startswith("deferred")]


def apply_all() -> dict[str, str]:
    """Apply every compatibility shim. Safe to call more than once.

    Shims that have already applied are not re-run -- some of them walk whole
    package trees, and the bootstrap may call this repeatedly while deferred
    shims wait for their targets.
    """
    for name, fn, deferrable in _SHIMS:
        if STATUS.get(name, "").startswith(_SETTLED):
            continue
        try:
            status = fn()
        except Exception as exc:  # never block startup on a shim
            STATUS[name] = f"FAILED: {exc}"
            logger.warning("RDNA3 compat shim %r failed: %s", name, exc)
            continue
        if deferrable and status.startswith("skipped"):
            # Not "not applicable" -- "not importable yet". Saying "skipped"
            # here is what let a shim silently never apply while doctor,
            # which runs after everything is imported, showed it as fine.
            status = "deferred" + status[len("skipped"):]
        STATUS[name] = status
    return STATUS
