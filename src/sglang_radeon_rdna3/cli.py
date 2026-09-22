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


def _check_sglang_import() -> tuple[str, str]:
    """Catch a `sglang/` directory in the cwd shadowing the real package.

    Python treats any directory named `sglang` as a namespace package, so
    cloning SGLang into your working directory makes `import sglang` resolve
    to the checkout root -- which has no modules in it. Everything installs
    cleanly and then `python -m sglang.launch_server` reports
    "No module named sglang.launch_server", which points nowhere useful.
    """
    import os

    try:
        import sglang
    except Exception as exc:
        return _BAD, f"cannot import sglang ({type(exc).__name__}: {exc})"

    path = getattr(sglang, "__file__", None)
    if path is None:
        where = ", ".join(getattr(sglang, "__path__", []) or ["?"])
        return _BAD, (
            f"'import sglang' resolved to the directory {where}, not to the "
            "installed package -- a folder named 'sglang' is shadowing it.\n"
            "        cd somewhere else, or move that directory."
        )

    # A console script puts its own bin/ first on sys.path, so this process is
    # immune to cwd shadowing even when `python -m sglang.launch_server` in
    # the same directory is not. Check the filesystem rather than our own
    # import, or we would cheerfully report success on a broken shell.
    shadow = os.path.join(os.getcwd(), "sglang")
    if os.path.isdir(shadow) and not os.path.samefile(
        shadow, os.path.dirname(path)
    ):
        return _WARN, (
            f"sglang imports fine here, but {shadow} will shadow it for\n"
            "        'python -m sglang.launch_server' run from this "
            "directory, which fails with\n"
            "        'No module named sglang.launch_server'. cd elsewhere "
            "before serving."
        )
    return _OK, f"sglang from {os.path.dirname(path)}"


def _check_serve_imports() -> tuple[str, str]:
    """Walk the import chain `sglang serve` walks. Nothing else can.

    The Dockerfile smoke test stops at `import sglang.multimodal_gen`, and it
    has to: everything past that point reaches for a GPU while merely being
    imported (kernels/ops/quantization/fp8_kernel.py calls
    torch.cuda.get_device_properties(0) at module scope), so a GPU-less builder
    cannot go further no matter how healthy the image is. That leaves the serve
    path -- the one users actually run -- unverified until runtime.

    It has already cost us once: an image shipped green with
    `ModuleNotFoundError: No module named 'aiter'` on exactly this chain. Here
    there is a real GPU, so there is no reason to stop.
    """
    try:
        import sglang.srt.layers.quantization  # noqa: F401
    except Exception as exc:
        return _BAD, (
            f"the quantization import chain is broken ({type(exc).__name__}: "
            f"{exc})\n"
            "        every model module imports it, so serving cannot start."
        )
    try:
        import sglang.cli.serve  # noqa: F401
    except Exception as exc:
        return _BAD, (
            f"cannot import the serve entrypoint ({type(exc).__name__}: {exc})"
        )
    return _OK, "serve import chain complete (quantization + cli.serve)"


def _check_platform() -> tuple[str, str]:
    try:
        from sglang.srt.utils.common import is_cuda, is_hip
    except Exception as exc:
        return _WARN, f"could not import sglang ({type(exc).__name__}: {exc})"
    if is_hip() and not is_cuda():
        return _OK, "sglang detects ROCm/HIP"
    return _BAD, f"unexpected platform: is_hip={is_hip()} is_cuda={is_cuda()}"


def _check_diffusion_platform() -> tuple[str, str]:
    """SGLang-Diffusion keeps its own platform detection, and it can be wrong.

    It does not ask torch. It probes for GPUs through amdsmi
    (platforms/__init__.py::rocm_platform_plugin) and falls back to
    CpuPlatform when that import fails -- which it does silently, because the
    probe is wrapped in `except Exception: logger.debug(...)`.

    The result is a diffusion runtime that believes it is on a CPU while
    sitting on a working GPU. It does not crash. It reports host RAM as
    available memory, and MiniMax-H3's denoise loop refuses to start because
    its platform gate wants CUDA or MPS. Nothing else in this report would
    catch it: torch, sgl_kernel and the LLM stack are all perfectly happy.
    """
    try:
        from sglang.multimodal_gen.runtime.platforms import current_platform
    except Exception as exc:
        return _WARN, f"diffusion platform unavailable ({type(exc).__name__}: {exc})"

    name = type(current_platform).__name__
    if current_platform.is_hip():
        return _OK, f"diffusion platform is {name}"
    try:
        import amdsmi  # noqa: F401
    except Exception:
        return _BAD, (
            f"diffusion resolved to {name}, not ROCm -- amdsmi is not "
            "installed, so its GPU probe found nothing.\n"
            "        pip install --no-deps /opt/rocm/share/amd_smi"
        )
    return _BAD, (
        f"diffusion resolved to {name}, not ROCm, despite amdsmi being "
        "installed -- its GPU probe found no devices."
    )


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
    results = (
        _check_sglang_import(),
        _check_sgl_kernel(),
        _check_platform(),
        _check_serve_imports(),
        _check_diffusion_platform(),
        _check_attention_backends(),
        _check_models(),
    )
    for status, msg in results:
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

    # Exit non-zero when something is actually broken. This is the image's
    # CMD, so the exit code is what an orchestrator sees: a doctor that always
    # reports success is not a doctor, and a container that starts cleanly on a
    # broken image is how a defect reaches production.
    broken = [msg for status, msg in results if status == _BAD]
    if not gpus:
        broken.append("no GPU visible to torch")
    if broken:
        print(f"\n  {len(broken)} problem(s) found -- see [{_BAD}] above.")
        return 1
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
