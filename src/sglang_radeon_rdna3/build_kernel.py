"""Build the ``sgl-kernel`` AoT extension for RDNA3.

Why this exists
---------------
SGLang's HIP path is not optional about ``sgl_kernel``:
``sglang/srt/layers/activation.py`` has a bare ``elif _is_hip:`` that imports
``silu_and_mul`` and friends from it, and that module is on the import path of
every model. Without a working ``sgl_kernel``, almost no model registers --
you will see 4 architectures instead of ~249.

The wheel published on PyPI as ``sglang-kernel`` is CUDA-only: it looks for
``libnvrtc.so.13`` and misreads gfx1100's ``major=11, minor=0`` as "CUDA
compute capability 110 / SM110". RDNA3 users must build from source, and the
upstream build script refuses to run for them.

This module applies a small set of reversible source patches (see
:mod:`sglang_radeon_rdna3.kernel_patches`) and runs the build. Beyond making
the build possible at all, the GPTQ patches enable 4-bit weights and so let a
27B model fit on a single 48 GB card.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sglang_radeon_rdna3 import kernel_patches
from sglang_radeon_rdna3.hardware import RDNA3_TARGETS


def find_aot_dir(sglang_src: Path) -> Path:
    for candidate in (
        sglang_src / "python" / "sglang" / "kernels" / "aot",
        sglang_src / "sgl-kernel",  # pre-relocation layout
    ):
        if (candidate / "setup_rocm.py").exists():
            return candidate
    raise FileNotFoundError(
        f"No setup_rocm.py under {sglang_src}. Point --sglang-src at a "
        "checkout of github.com/sgl-project/sglang."
    )


def detect_target() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        pass
    return "gfx1100"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sglang-rdna3 build-kernel",
        description="Patch and build sgl-kernel for RDNA3 (gfx110x).",
    )
    ap.add_argument("--sglang-src", type=Path, required=True,
                    help="Path to a sglang source checkout.")
    ap.add_argument("--target", default=None,
                    help="gfx target (default: autodetect, else gfx1100).")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--install", action="store_true",
                    help="pip install the wheel after building.")
    ap.add_argument("--patch-only", action="store_true",
                    help="Apply the source patches and stop.")
    ap.add_argument("--no-gptq", action="store_true",
                    help="Skip the GPTQ patches (base RDNA3 support only).")
    args = ap.parse_args(argv)

    target = args.target or detect_target()
    if target not in RDNA3_TARGETS:
        print(f"warning: {target} is not an RDNA3 target {RDNA3_TARGETS}; "
              "continuing anyway.", file=sys.stderr)

    aot = find_aot_dir(args.sglang_src)
    patches = kernel_patches.BASE_PATCHES
    if not args.no_gptq:
        patches = patches + kernel_patches.GPTQ_PATCHES

    print(f"[1/3] build dir: {aot}")
    results = kernel_patches.apply(aot, patches)
    for line in results:
        print(f"      {line}")
    if any("ANCHOR NOT FOUND" in r for r in results):
        print("\nrefusing to build against unrecognised sources: a missing "
              "anchor means upstream moved, and a partial patch can produce a "
              "kernel that builds but misbehaves.\n"
              "Re-run with --no-gptq, or open an issue with your sglang "
              "commit.", file=sys.stderr)
        return 2
    if args.patch_only:
        return 0

    # kernels_data is imported by the build but is not declared as a build
    # requirement upstream; installing it avoids a confusing traceback.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q", "kernels_data"],
        check=False,
    )

    # Stale hipify output for files we patched would mask our edits. Be
    # surgical: some .hip files in this tree are hand-written sources listed
    # in the build (csrc/allreduce/*.hip), and removing those breaks the
    # build. Only drop the generated counterparts of files we actually
    # touched.
    for patch in patches:
        if patch.relpath.endswith(".cu"):
            generated = (aot / patch.relpath).with_suffix(".hip")
            if generated.exists():
                generated.unlink()

    env = {**os.environ, "AMDGPU_TARGET": target, "MAX_JOBS": str(args.jobs)}
    print(f"[2/3] building for {target} with {args.jobs} jobs "
          "(a few minutes)...")
    rc = subprocess.run(
        [sys.executable, "setup_rocm.py", "bdist_wheel"], cwd=aot, env=env
    ).returncode
    if rc != 0:
        print("build failed", file=sys.stderr)
        return rc

    wheels = sorted((aot / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print("build reported success but produced no wheel", file=sys.stderr)
        return 1
    wheel = wheels[-1]
    print(f"[3/3] wheel: {wheel}")

    if args.install:
        # --force-reinstall matters: a CUDA sglang-kernel may already be here.
        rc = subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "--force-reinstall", str(wheel),
        ]).returncode
        if rc == 0:
            print("installed. verify with: sglang-rdna3 doctor")
        return rc

    print(f"install it with: pip install --no-deps --force-reinstall {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
