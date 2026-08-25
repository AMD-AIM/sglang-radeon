"""``sglang-rdna3`` command line interface."""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import sys

from sglang_radeon_rdna3 import __version__, compat
from sglang_radeon_rdna3.hardware import detect_gpus, recommended_server_args

_OK = "ok  "
_WARN = "warn"
_BAD = "FAIL"


def _check_sgl_kernel() -> tuple[str, str]:
    """The most common failure by far: a CUDA sgl_kernel, or none at all.

    This bites twice. First when nothing is installed, and again later --
    `sglang-kernel` is listed in SGLang's own dependencies, so a routine
    `pip install -r` can silently drop a CUDA wheel on top of the gfx1100
    kernel you built. We therefore check the binary, not just the import.
    """
    import glob
    import os
    import subprocess

    try:
        import sgl_kernel  # noqa: F401
    except ImportError as exc:
        msg = str(exc)
        if "libnvrtc" in msg or "sm100" in msg or "common_ops" in msg:
            return _BAD, (
                "sgl_kernel is installed but is a CUDA build. This usually "
                "means pip pulled the `sglang-kernel` wheel from PyPI over "
                "your local build. Fix:\n"
                "        pip uninstall -y sglang-kernel\n"
                "        sglang-rdna3 build-kernel --sglang-src <path> --install"
            )
        return _BAD, (
            "sgl_kernel is missing. SGLang's HIP path imports it "
            "unconditionally, so no models will register:\n"
            "        sglang-rdna3 build-kernel --sglang-src <path> --install"
        )

    pkg_dir = os.path.dirname(sgl_kernel.__file__)
    sos = glob.glob(os.path.join(pkg_dir, "common_ops*.so"))
    if not sos:
        return _WARN, f"sgl_kernel imports, but no common_ops*.so in {pkg_dir}"

    # Confirm the binary really contains code for this GPU.
    try:
        out = subprocess.run(["strings", sos[0]], capture_output=True,
                             text=True, timeout=60).stdout
        targets = sorted(set(re.findall(r"gfx\d+", out)))
    except Exception:
        targets = []

    if not targets:
        return _WARN, (
            f"sgl_kernel loaded from {pkg_dir}, but no gfx target found in the "
            "binary (is `strings` available?)"
        )

    gpus = detect_gpus()
    want = {g.gfx for g in gpus}
    if want and not (want & set(targets)):
        return _BAD, (
            f"sgl_kernel was built for {', '.join(targets)} but this machine "
            f"has {', '.join(sorted(want))}. Rebuild:\n"
            "        sglang-rdna3 build-kernel --sglang-src <path> --install"
        )
    return _OK, f"sgl_kernel built for {', '.join(targets)}"


def _check_platform() -> tuple[str, str]:
    try:
        from sglang.srt.utils.common import is_cuda, is_hip
    except Exception as exc:
        return _WARN, f"could not import sglang ({type(exc).__name__}: {exc})"
    if is_hip() and not is_cuda():
        return _OK, "sglang detects ROCm/HIP"
    return _BAD, f"unexpected platform: is_hip={is_hip()} is_cuda={is_cuda()}"


def _check_attention_backends() -> tuple[str, str]:
    try:
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )
    except Exception as exc:
        return _WARN, f"registry unavailable ({exc})"
    if "triton" in ATTENTION_BACKENDS:
        return _OK, f"{len(ATTENTION_BACKENDS)} backends, triton available"
    return _BAD, "triton backend missing -- no usable attention backend"


def _check_models() -> tuple[str, str]:
    try:
        from sglang.srt.models.registry import ModelRegistry

        n = len(ModelRegistry.models)
    except Exception as exc:
        return _WARN, f"model registry unavailable ({exc})"
    if n > 100:
        return _OK, f"{n} model architectures registered"
    return _BAD, (
        f"only {n} architectures registered -- the import chain is broken, "
        "usually a missing or CUDA-only sgl_kernel"
    )


def cmd_doctor(_args) -> int:
    print(f"sglang-radeon-rdna3 {__version__}\n")

    gpus = detect_gpus()
    if not gpus:
        print(f"  [{_BAD}] no GPU visible to torch")
    for g in gpus:
        tag = _OK if g.is_rdna3 else _WARN
        note = "" if g.is_rdna3 else "  (not RDNA3; this package targets gfx110x)"
        print(f"  [{tag}] GPU {g.index}: {g.name} {g.gfx} "
              f"{g.total_memory_gib} GiB, {g.compute_units} CUs{note}")

    print()
    for status, msg in (
        _check_sgl_kernel(),
        _check_platform(),
        _check_attention_backends(),
        _check_models(),
    ):
        print(f"  [{status}] {msg}")

    print("\n  compat shims:")
    for name, state in (compat.STATUS or compat.apply_all()).items():
        print(f"    - {name}: {state}")

    if gpus:
        print("\n  suggested flags for a 27B BF16 model:")
        rec = recommended_server_args(55.0, gpus)
        print("    " + " ".join(
            f"--{k.replace('_', '-')}" if v is True
            else f"--{k.replace('_', '-')} {v}"
            for k, v in rec.items()
        ))
    return 0


def cmd_serve_args(args) -> int:
    """Print a ready-to-run launch_server command."""
    gpus = detect_gpus()
    rec = recommended_server_args(args.model_size_gib, gpus)
    if args.tp_size:
        rec["tp_size"] = args.tp_size
    flags = " ".join(
        f"--{k.replace('_', '-')}" if v is True
        else f"--{k.replace('_', '-')} {v}"
        for k, v in rec.items()
    )
    print(
        f"python -m sglang.launch_server --model-path {shlex.quote(args.model)} "
        f"{flags} --host 0.0.0.0 --port 30000"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sglang-rdna3",
        description="Run SGLang on AMD Radeon RDNA3 (gfx110x) GPUs.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check this environment end to end")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("serve-args", help="print a recommended launch command")
    s.add_argument("model", help="model path or HF repo id")
    s.add_argument("--model-size-gib", type=float, default=55.0)
    s.add_argument("--tp-size", type=int, default=None)
    s.set_defaults(func=cmd_serve_args)

    b = sub.add_parser("build-kernel", help="patch and build sgl-kernel")
    b.set_defaults(func=None)

    args, rest = ap.parse_known_args(argv)
    if args.cmd == "build-kernel":
        mod = importlib.import_module("sglang_radeon_rdna3.build_kernel")
        return mod.main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
