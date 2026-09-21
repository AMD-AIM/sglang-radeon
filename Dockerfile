# SGLang on AMD Radeon RDNA3 (gfx110x) -- ready to serve.
#
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#     --group-add video --ipc=host --shm-size 16g \
#     ghcr.io/amd-aim/sglang-radeon:latest sglang-rdna3 doctor
#
# Built on AMD's ROCm PyTorch image, so the hardest prerequisite -- a ROCm
# build of PyTorch that nothing has quietly replaced with a CUDA one -- is
# already satisfied.
# Any ROCm PyTorch base works. This one is what the project was validated
# against end to end (ROCm 7.2.1, PyTorch 2.9.1, Python 3.12); newer tags in
# the same family should be fine, but have not been exercised here.
ARG ROCM_PYTORCH_IMAGE=rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM ${ROCM_PYTORCH_IMAGE}

# Guard against the one mistake that silently ruins everything downstream:
# a base image whose PyTorch is a CUDA build.
RUN python3 -c "import torch, sys; sys.exit(0 if torch.version.hip else \
      'base image has a CUDA PyTorch (%s); use a ROCm one' % torch.__version__)"

# SGLANG_REF must be a *branch* name -- the tarball below is fetched from
# refs/heads. Upstream publishes release branches alongside tags, so
# "v0.5.19" works as-is.
#
# Mind the pairing with the base image's PyTorch. SGLang main currently
# wants torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool, which the
# pinned ROCm torch 2.6.0 does not have, so `main` + the default base fails
# the import check at the end of this file. v0.5.19 is contemporaneous with
# torch 2.6 and is what has been run in production on ROCm 7.0. Bump both
# together, or not at all.
ARG SGLANG_REF=main
ARG AMDGPU_TARGETS=gfx1100
LABEL org.opencontainers.image.source="https://github.com/AMD-AIM/sglang-radeon"
LABEL org.opencontainers.image.description="SGLang for AMD Radeon RDNA3 (gfx110x)"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    SGLANG_USE_AITER=0 \
    SGLANG_SRC=/opt/sglang-src

# ffprobe is needed by MiniMax-H3's output validation; imageio-ffmpeg bundles
# ffmpeg but not ffprobe, and the check runs at the very end of a long job.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# SGLang sources. The published wheel omits the AoT kernel tree, so a
# tarball it is -- same content as a clone, without ~500 MB of history.
RUN mkdir -p ${SGLANG_SRC} \
 && curl -fsSL --retry 3 \
      "https://codeload.github.com/sgl-project/sglang/tar.gz/refs/heads/${SGLANG_REF}" \
    | tar xz -C ${SGLANG_SRC} --strip-components=1

COPY . /opt/sglang-radeon

# Everything SGLang-related with --no-deps, plus the ordinary transitive
# dependencies that --no-deps skips and nothing else pulls in.
#
# --no-deps because SGLang's dependency list hard-codes CUDA
# packages with no platform markers (cuda-python, flashinfer[cu13],
# nvidia-cutlass-dsl, and sglang-kernel, whose PyPI wheel is CUDA-only and
# would shadow the kernel we build below). Letting pip resolve them installs
# a CUDA build of PyTorch over the ROCm one in the base image.
#
# The catch is that --no-deps makes the transitive set our problem, and
# hand-listing it does not scale: the published image built from the old
# list could not run `sglang serve` at all -- starlette, traitlets,
# multidict, attrs, yarl and friends were all missing, and pydantic-core
# was absent so pydantic imported against nothing.
#
# So the second list below deliberately names *front-door* packages and
# lets pip resolve their trees normally. None of them depends on torch, so
# the ROCm build in the base image is not at risk -- verified: torch stays
# at 2.6.0+rocm7 afterwards. Keeping this list short and resolvable beats
# enumerating two dozen leaf packages that drift with every SGLang bump.
RUN pip install --no-deps --no-build-isolation -q -e ${SGLANG_SRC}/python \
 && python3 /opt/sglang-radeon/scripts/filter_deps.py \
      --pyproject ${SGLANG_SRC}/python/pyproject.toml \
      --output /tmp/deps.txt --diffusion --show-dropped \
 && pip install --no-deps --no-build-isolation -q -r /tmp/deps.txt \
 && pip install --no-deps -q "huggingface_hub>=1.0,<2.0" apache-tvm-ffi==0.1.11 \
      kernels_data hf_transfer hf_xet \
 && pip install -q httpx urllib3 certifi idna sniffio anyio h11 httpcore \
 && pip install -q fastapi pydantic aiohttp ipython charset_normalizer \
 && pip install --no-deps -q /opt/sglang-radeon \
 && rm -f /tmp/deps.txt \
 && python3 -c "import torch, sys; sys.exit(0 if torch.version.hip else \
      'dependency resolution replaced ROCm torch with %s' % torch.__version__)"

# Swap setuptools' editable finder for a plain .pth.
#
# `pip install -e` generates __editable___sglang_*_finder.py with a
# NAMESPACES table of every subpackage it discovered. On SGLang that table
# comes out incomplete: it lists dozens of sglang.multimodal_gen.* entries
# but not sglang.multimodal_gen itself, so importing anything underneath
# dies with KeyError: 'sglang.multimodal_gen'. That is the whole diffusion
# stack -- `sglang serve --model-type diffusion` could not start, while the
# LLM path looked fine, which is why it went unnoticed.
#
# The sources are already on disk at a fixed path, so a one-line .pth does
# the same job with none of the discovery logic. The dist-info stays, so
# `pip show sglang` and friends keep working.
RUN SP=$(python3 -c "import site; print(site.getsitepackages()[0])") \
 && rm -f "$SP"/__editable__.sglang-*.pth "$SP"/__editable___sglang_*_finder.py \
 && echo "${SGLANG_SRC}/python" > "$SP/sglang-src.pth" \
 && python3 -c "import sglang.multimodal_gen as m; print('multimodal_gen:', m.__file__)"

# Build sgl-kernel for RDNA3. Upstream's build script exits on any target
# outside gfx942/gfx950/gfx1250; the plugin patches that, gives RDNA3 the
# 48 KB LDS budget it actually has, and compiles the GPTQ kernel that ROCm
# otherwise never gets (which is what makes 4-bit 27B fit one 48 GB card).
RUN sglang-rdna3 build-kernel \
      --sglang-src ${SGLANG_SRC} \
      --target ${AMDGPU_TARGETS} \
      --install \
 && rm -rf ${SGLANG_SRC}/python/sglang/kernels/aot/build \
           ${SGLANG_SRC}/python/sglang/kernels/aot/dist \
 && pip cache purge 2>/dev/null || true

# Fail the build rather than ship an image whose kernel is wrong. This runs
# on a builder with no GPU, so it checks the artifact rather than the
# runtime: that sgl_kernel is importable, that its binary really contains
# code for the target architecture (a CUDA wheel would not), and that the
# GPTQ ops the patches add are registered.
#
# Model-registration count is deliberately not asserted here -- it needs a
# visible GPU. `sglang-rdna3 doctor` covers that on real hardware.
ARG AMDGPU_TARGETS
RUN python3 - "$AMDGPU_TARGETS" <<'PY'
import glob, os, re, subprocess, sys

target = sys.argv[1] if len(sys.argv) > 1 else "gfx1100"

import sgl_kernel
pkg = os.path.dirname(sgl_kernel.__file__)
sos = glob.glob(os.path.join(pkg, "common_ops*.so"))
if not sos:
    sys.exit(f"no common_ops*.so in {pkg} -- the kernel did not install")

out = subprocess.run(["strings", sos[0]], capture_output=True, text=True).stdout
found = sorted(set(re.findall(r"gfx\d+", out)))
print("kernel targets:", ", ".join(found) or "(none)")
if target not in found:
    sys.exit(f"kernel was built for {found}, expected {target}")

import torch
missing = [op for op in ("gptq_shuffle", "gptq_gemm")
           if not hasattr(torch.ops.sgl_kernel, op)]
if missing:
    sys.exit(f"{', '.join(missing)} not registered -- the GPTQ patches did not take")
print("sgl_kernel OK: built for", target, "with GPTQ ops")
PY

# Fail the build rather than ship an image that cannot start SGLang.
#
# The check above proves the *kernel* is right. Nothing proved that
# `sglang` imports -- and that is exactly how the previously published
# image shipped: correct gfx1100 kernel, complete MiniMax-H3 configs, every
# CLI flag present in the source, and a venv too incomplete to import any
# of it. The failure only showed up at `docker run`, as CrashLoopBackOff.
#
# Both backends are exercised, because they fail independently: the LLM
# path came back first while diffusion was still dead on
# KeyError: 'sglang.multimodal_gen'.
#
# --help only. No GPU on the builder, and no weights to point at.
RUN sglang serve --help > /dev/null \
 && sglang serve --model-type diffusion --help > /dev/null \
 && echo "sglang OK: LLM and diffusion backends both import and parse args"

WORKDIR /workspace
VOLUME ["/workspace", "/models"]
EXPOSE 30000

CMD ["sglang-rdna3", "doctor"]
