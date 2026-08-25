#!/usr/bin/env python3
"""Emit SGLang's dependency list with the CUDA-only entries removed.

Why this is needed
------------------
SGLang declares CUDA packages as unconditional core dependencies, with no
platform markers:

    cuda-python>=13.0
    cuda-tile==1.6.0rc5
    flashinfer_python[cu13]==0.6.17
    nvidia-cutlass-dsl[cu13]==4.6.2
    sglang-kernel==0.4.6.post1

On a ROCm machine, resolving those installs a CUDA build of PyTorch over the
working one -- silently, since pip considers it a normal upgrade. Filtering by
name alone is not enough either: several packages pull CUDA torch in
transitively (humming-kernels[cu13] is the one that caught us out), so
callers must also install with --no-deps.

`sglang-kernel` deserves special mention: the PyPI wheel is CUDA-only and
will shadow a locally built gfx1100 kernel, dropping model registration from
~249 architectures to 4.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib

#: Dropped because they are CUDA-only, or because they would replace a
#: working ROCm install.
EXCLUDE = {
    # CUDA runtime and kernel libraries
    "cuda-python",
    "cuda-tile",
    "flashinfer-python",
    "flashinfer_python",
    "humming-kernels",
    "nvidia-cutlass-dsl",
    "nvidia-mathdx",
    "nvidia-ml-py",
    "nvidia-modelopt",
    # CUDA-only wheel that shadows the locally built gfx1100 kernel
    "sglang-kernel",
    # Provided by the ROCm install; never let pip re-resolve these
    "torch",
    "torchvision",
    "torchaudio",
    "torch-c-dlpack-ext",
    "triton",
    # Pulled in explicitly by the installer instead (needed, not CUDA-only)
    "apache-tvm-ffi",
}

#: Additional exclusions for the diffusion extras.
EXCLUDE_DIFFUSION = {
    "nvidia-modelopt",
    "st-attn",
    "st_attn",
    "vsa",
    "runai-model-streamer",
    "runai_model_streamer",
}

#: Extras like [cu13] drag CUDA wheels in through the back door.
_CUDA_EXTRA = re.compile(r"\[cu\d+\]")


def requirement_name(spec: str) -> str:
    """Package name from a requirement string, normalised for comparison."""
    return re.split(r"[\[<>=!;\s]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def filter_requirements(specs, exclude) -> list[str]:
    return [
        _CUDA_EXTRA.sub("", spec)
        for spec in specs
        if requirement_name(spec) not in exclude
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pyproject", required=True,
                    help="Path to sglang/python/pyproject.toml")
    ap.add_argument("--output", help="Write here instead of stdout")
    ap.add_argument("--diffusion", action="store_true",
                    help="Also emit the diffusion extras (for MiniMax-H3)")
    ap.add_argument("--show-dropped", action="store_true")
    args = ap.parse_args(argv)

    with open(args.pyproject, "rb") as fh:
        project = tomllib.load(fh)["project"]

    exclude = set(EXCLUDE)
    specs = list(project["dependencies"])
    if args.diffusion:
        extras = project.get("optional-dependencies", {}).get("diffusion", [])
        specs += extras
        exclude |= EXCLUDE_DIFFUSION

    kept = filter_requirements(specs, exclude)
    dropped = [s for s in specs if requirement_name(s) in exclude]

    if args.show_dropped:
        print("dropped:", ", ".join(sorted(dropped)), file=sys.stderr)

    text = "\n".join(kept) + "\n"
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"{len(kept)} requirements -> {args.output} "
              f"({len(dropped)} dropped)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
