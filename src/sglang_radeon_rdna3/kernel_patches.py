"""Source patches applied to sgl-kernel before building it for RDNA3.

Each patch is a small, reversible edit anchored on an exact string. If
upstream changes the surrounding code the anchor stops matching and we report
it loudly rather than silently producing a half-patched kernel.

Every patch here is also a candidate fix for SGLang itself; see
docs/upstream-issues.md for the write-ups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class Patch:
    """One find/replace against a file relative to the AoT build directory.

    ``is_applied`` decides idempotency. It cannot be inferred from ``new``
    alone: some patches keep their anchor inside ``new`` (append-style), and
    others have a ``new`` that is a substring of ``old`` (keeping one arm of
    an ``#ifdef``). Both cases misreport if you just test for ``new``, so each
    patch states its own check.
    """

    name: str
    relpath: str
    old: str
    new: str
    why: str
    is_applied: Callable[[str], bool]


def _contains(needle: str) -> Callable[[str], bool]:
    return lambda text: needle in text


def _lacks(needle: str) -> Callable[[str], bool]:
    return lambda text: needle not in text


# ---------------------------------------------------------------------------
# 1. Architecture allowlist. Upstream sys.exit(1)s on anything outside CDNA.
# ---------------------------------------------------------------------------
ARCH_GATE = Patch(
    name="arch_allowlist",
    relpath="setup_rocm.py",
    old='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:',
    new=('if amdgpu_target not in ["gfx942", "gfx950", "gfx1250", '
         '"gfx1100", "gfx1101", "gfx1102"]:'),
    why="the build refuses to run for any non-CDNA target",
    is_applied=_contains('"gfx1100", "gfx1101", "gfx1102"]:'),
)

# ---------------------------------------------------------------------------
# 2. Shared-memory budget. RDNA3 has 64 KB of LDS per workgroup, the same as
#    gfx942, so it needs the 48 KB budget -- not the 128 KB one upstream gives
#    every non-gfx942 target. Patching only the allowlist yields a kernel
#    configured for shared memory the hardware does not have.
# ---------------------------------------------------------------------------
SMEM_BUDGET = Patch(
    name="lds_budget",
    relpath="setup_rocm.py",
    old=('topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target == "gfx942" '
         "else 32 * 1024 * 4"),
    new=('_LDS_64KB_ARCHS = ("gfx942", "gfx1100", "gfx1101", "gfx1102")\n'
         "topk_dynamic_smem_bytes = (\n"
         "    48 * 1024 if amdgpu_target in _LDS_64KB_ARCHS else 32 * 1024 * 4\n"
         ")"),
    why="RDNA3 LDS is 64KB/workgroup; the 128KB budget is wrong for it",
    is_applied=_contains("_LDS_64KB_ARCHS"),
)

# ---------------------------------------------------------------------------
# 3-4. GPTQ support. The kernel source is excluded from the ROCm build and the
#      ops are registered only in the CUDA extension, so 4-bit models fail:
#        '_OpNamespace' 'sgl_kernel' object has no attribute 'gptq_shuffle'
#      This is what makes a 27B model fit -- and run faster -- on one 48 GB
#      card than split across two in BF16.
# ---------------------------------------------------------------------------
GPTQ_SOURCE = Patch(
    name="gptq_source",
    relpath="setup_rocm.py",
    old='    "csrc/elementwise/pos_enc.cu",\n]',
    new=('    "csrc/elementwise/pos_enc.cu",\n'
         '    "csrc/gemm/gptq/gptq_kernel.cu",\n]'),
    why="the GPTQ kernel is never compiled for ROCm",
    is_applied=_contains("csrc/gemm/gptq/gptq_kernel.cu"),
)

GPTQ_REGISTER = Patch(
    name="gptq_register",
    relpath="csrc/common_extension_rocm.cc",
    old='  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");',
    new=('  /*\n   * From csrc/gemm/gptq\n   */\n'
         '  m.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");\n'
         '  m.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);\n\n'
         '  m.def("gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, '
         'Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, int bit) -> Tensor");\n'
         '  m.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);\n\n'
         '  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");'),
    why="gptq ops are registered only in the CUDA extension file",
    # `new` re-includes the anchor, so detect the inserted registration.
    is_applied=_contains("gptq_shuffle"),
)

# ---------------------------------------------------------------------------
# 5-6. Stale USE_ROCM branches in the GPTQ kernel. Upstream already carries
#      ROCm paths, written when HIP exposed half2 lanes as `unsigned short`.
#      In ROCm 7.x they are `__half`, matching CUDA, so those branches fail:
#          no known conversion from '__half' to 'unsigned short'
# ---------------------------------------------------------------------------
GPTQ_HALF2_ZERO = Patch(
    name="gptq_half2_zero_init",
    relpath="csrc/gemm/gptq/gptq_kernel.cu",
    old=("#ifndef USE_ROCM\n"
         "      res2 = {};\n"
         "#else\n"
         "      res2.x = __half_as_ushort(__float2half(0));\n"
         "      res2.y = __half_as_ushort(__float2half(0));\n"
         "#endif"),
    # Built explicitly: `res2 = {}` is ambiguous against HIP __half2's
    # several operator= overloads.
    new=("      // ROCm 7.x half2 lanes are __half, matching CUDA. Built\n"
         "      // explicitly because `= {}` is ambiguous against HIP's\n"
         "      // several __half2 assignment overloads.\n"
         "      res2 = __halves2half2(__float2half(0.f), __float2half(0.f));"),
    why="the USE_ROCM branch targets a pre-7.x HIP half2 layout",
    is_applied=_lacks("__half_as_ushort"),
)

GPTQ_HALF2_ADD = Patch(
    name="gptq_half2_accumulate",
    relpath="csrc/gemm/gptq/gptq_kernel.cu",
    old=("#ifndef USE_ROCM\n"
         "      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));\n"
         "#else\n"
         "      res[m] = __hadd(res[m], __hadd(__ushort_as_half(res2.x), "
         "__ushort_as_half(res2.y)));\n"
         "#endif"),
    new="      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));",
    why="same stale half2 assumption, on the accumulate path",
    # `new` is a substring of `old` (it is the CUDA arm), so test for the
    # absence of the ROCm helper instead.
    is_applied=_lacks("__ushort_as_half"),
)

BASE_PATCHES = (ARCH_GATE, SMEM_BUDGET)
GPTQ_PATCHES = (GPTQ_SOURCE, GPTQ_REGISTER, GPTQ_HALF2_ZERO, GPTQ_HALF2_ADD)


def apply(aot_dir: Path, patches: Iterable[Patch], backup: bool = True) -> list[str]:
    """Apply ``patches`` under ``aot_dir``; returns one status line each."""
    results: list[str] = []
    by_file: dict[str, list[Patch]] = {}
    for patch in patches:
        by_file.setdefault(patch.relpath, []).append(patch)

    for relpath, group in by_file.items():
        path = aot_dir / relpath
        if not path.exists():
            results.extend(f"{p.name}: MISSING file {relpath}" for p in group)
            continue

        text = original = path.read_text()
        for patch in group:
            if patch.is_applied(text):
                results.append(f"{patch.name}: already applied")
            elif patch.old in text:
                text = text.replace(patch.old, patch.new)
                results.append(f"{patch.name}: applied")
            else:
                results.append(
                    f"{patch.name}: ANCHOR NOT FOUND -- upstream changed; "
                    f"please report ({patch.why})"
                )

        if text != original:
            if backup:
                bak = path.with_suffix(path.suffix + ".orig")
                if not bak.exists():
                    bak.write_text(original)
            path.write_text(text)

    return results
