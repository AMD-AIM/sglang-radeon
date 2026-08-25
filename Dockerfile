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

# Everything with --no-deps: SGLang's dependency list hard-codes CUDA
# packages with no platform markers (cuda-python, flashinfer[cu13],
# nvidia-cutlass-dsl, and sglang-kernel, whose PyPI wheel is CUDA-only and
# would shadow the kernel we build below). Letting pip resolve them installs
# a CUDA build of PyTorch over the ROCm one in the base image.
RUN pip install --no-deps --no-build-isolation -q -e ${SGLANG_SRC}/python \
 && python3 /opt/sglang-radeon/scripts/filter_deps.py \
      --pyproject ${SGLANG_SRC}/python/pyproject.toml \
      --output /tmp/deps.txt --diffusion --show-dropped \
 && pip install --no-deps --no-build-isolation -q -r /tmp/deps.txt \
 && pip install --no-deps -q "huggingface_hub>=1.0,<2.0" apache-tvm-ffi==0.1.11 \
      kernels_data hf_transfer hf_xet \
 && pip install --no-deps -q /opt/sglang-radeon \
 && rm -f /tmp/deps.txt

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

# Fail the build rather than ship an image whose kernel does not load. A
# CUDA sgl_kernel shows up as ~4 registered architectures instead of ~249.
RUN python3 - <<'PY'
import sys
from sglang.srt.models.registry import ModelRegistry
import sglang.srt.models.qwen3_5  # noqa: F401  (force discovery)
n = len(ModelRegistry.models)
print(f"model architectures registered: {n}")
if n < 100:
    sys.exit(f"only {n} architectures registered -- the kernel is not loading")
import sgl_kernel, torch
for op in ("gptq_shuffle", "gptq_gemm"):
    if not hasattr(torch.ops.sgl_kernel, op):
        sys.exit(f"{op} missing -- the GPTQ patches did not take")
print("sgl_kernel OK, GPTQ ops present")
PY

WORKDIR /workspace
VOLUME ["/workspace", "/models"]
EXPOSE 30000

CMD ["sglang-rdna3", "doctor"]
