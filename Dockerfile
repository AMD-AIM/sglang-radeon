# SGLang on AMD Radeon RDNA3 (gfx110x) -- ready to serve.
#
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#     --group-add video --ipc=host --shm-size 16g \
#     ghcr.io/amd-aim/sglang-radeon:latest sglang-rdna3 doctor
#
# Built on AMD's ROCm PyTorch image, so the hardest prerequisite -- a ROCm
# build of PyTorch that nothing has quietly replaced with a CUDA one -- is
# already satisfied.
#
# The tag matters more than it looks. SGLang reaches into private CUDA
# allocator APIs (_cuda_beginAllocateCurrentThreadToPool and friends) that
# landed in torch 2.8, so anything older cannot import SGLang at all -- and
# the symptom is a KeyError deep in an unrelated import, not a clear message
# about torch. This is the tag README.md reports as validated (ROCm 7.2.1,
# PyTorch 2.9.1, Python 3.12, HIP 7.2.53211); newer tags in the same family
# work too, older ones do not. Do not lower it to "match" an older SGLang.
ARG ROCM_PYTORCH_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${ROCM_PYTORCH_IMAGE}

# Guard against the one mistake that silently ruins everything downstream:
# a base image whose PyTorch is a CUDA build.
RUN python3 -c "import torch, sys; sys.exit(0 if torch.version.hip else \
      'base image has a CUDA PyTorch (%s); use a ROCm one' % torch.__version__)"

# SGLANG_REF must be a *branch* name -- the tarball below is fetched from
# refs/heads. Upstream publishes release branches alongside tags, so
# "v0.5.19" works as-is.
#
# The two refs are not interchangeable, and neither is strictly better:
#
#   main      diffusion works (`import sglang.multimodal_gen` succeeds), but
#             the AoT GPTQ kernel tree is gone -- upstream deleted it, so
#             4-bit GPTQ models cannot run on the resulting image.
#   v0.5.19   has the AoT GPTQ tree, but `import sglang.multimodal_gen`
#             fails on `No module named 'aiter'` (aiter is CDNA-only), even
#             with a GPU present. See docs/upstream-issues.md.
#
# main is the default because diffusion is the workload this image exists
# for. Switch to v0.5.19 only if you need 4-bit GPTQ and not diffusion.
#
# One thing this ref is *not* for: working around import errors. If SGLang
# stops importing, check the base image's torch version first -- an old
# torch presents as an SGLang bug, and pinning SGLang back to match it
# treats the symptom.
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

# Everything SGLang-related with --no-deps, then close the transitive gap
# that --no-deps leaves behind.
#
# --no-deps because SGLang's dependency list hard-codes CUDA packages with
# no platform markers (cuda-python, flashinfer[cu13], nvidia-cutlass-dsl,
# and sglang-kernel, whose PyPI wheel is CUDA-only and would shadow the
# kernel we build below). It is not merely preferable here, it is the only
# option: the set is unresolvable against a ROCm base by construction --
# compressed-tensors pins torch>=2.10.0 and SGLang itself pins torch==2.13.0,
# so a full resolve ends in ResolutionImpossible, and the only way to make
# it resolvable is to let pip install the CUDA torch it is asking for.
#
# The catch is that --no-deps makes the transitive set our problem, and
# hand-listing it does not scale. The list that used to live here drifted
# until the published image could not run `sglang serve` at all -- starlette,
# traitlets, multidict, attrs, yarl and pydantic-core were all missing --
# and adding names one at a time only moves the failure: repair the venv
# enough to import transformers and the next build dies on `regex`.
#
# So close_deps.py computes the closure from the installed distributions'
# own metadata instead of guessing it. See its docstring for why a clean
# `pip check` is deliberately *not* the goal.
#
# httpx goes in first, before anything else: this base ships `kernels` +
# `huggingface_hub`, whose import chain needs httpx, and SGLang's build hook
# imports them while generating its metadata. Install it afterwards and the
# editable install above never gets far enough to need it.
RUN pip install -q httpx \
 && pip install --no-deps --no-build-isolation -q -e ${SGLANG_SRC}/python \
 && python3 /opt/sglang-radeon/scripts/filter_deps.py \
      --pyproject ${SGLANG_SRC}/python/pyproject.toml \
      --output /tmp/deps.txt --diffusion --show-dropped \
 && pip install --no-deps --no-build-isolation -q -r /tmp/deps.txt \
 && pip install --no-deps -q "huggingface_hub>=1.0,<2.0" apache-tvm-ffi==0.1.11 \
      kernels_data hf_transfer hf_xet \
 && pip install --no-deps -q /opt/sglang-radeon \
 && python3 /opt/sglang-radeon/scripts/close_deps.py \
 && rm -f /tmp/deps.txt \
 && python3 -c "import torch, sys; sys.exit(0 if torch.version.hip else \
      'dependency resolution replaced ROCm torch with %s' % torch.__version__)"

# Build sgl-kernel for RDNA3. Upstream's build script exits on any target
# outside gfx942/gfx950/gfx1250; the plugin patches that and gives RDNA3 the
# same 48 KB LDS budget as gfx942. On sglang <= v0.5.19 it also compiles the
# GPTQ kernel that ROCm otherwise never gets (what makes 4-bit 27B fit one
# 48 GB card); newer sglang has no AoT GPTQ tree and build-kernel says so
# and builds the rest.
#
# The braces are load-bearing. `a && b && c || true` binds as
# `(a && b && c) || true`, so the trailing `|| true` -- meant only to excuse
# an old pip without `cache purge` -- swallowed a failed kernel build, and
# the image went on to the verification step below with no sgl_kernel in it.
# That is how a build that "succeeded" in 1.2 seconds reported success.
RUN sglang-rdna3 build-kernel \
      --sglang-src ${SGLANG_SRC} \
      --target ${AMDGPU_TARGETS} \
      --install \
 && rm -rf ${SGLANG_SRC}/python/sglang/kernels/aot/build \
           ${SGLANG_SRC}/python/sglang/kernels/aot/dist \
 && { pip cache purge 2>/dev/null || true; }

# Fail the build rather than ship an image whose kernel is wrong. This runs
# on a builder with no GPU, so it checks the artifact rather than the
# runtime: that sgl_kernel is importable, that its binary really contains
# code for the target architecture (a CUDA wheel would not), and that the
# GPTQ ops the patches add are registered.
#
# Model-registration count is deliberately not asserted here -- it needs a
# visible GPU. `sglang-rdna3 doctor` covers that on real hardware.
#
# The GPTQ assertion is conditional on the source tree, not unconditional.
# Upstream removed the AoT GPTQ tree after v0.5.19, and build-kernel skips
# those patches when it is absent; asserting the ops regardless would fail
# the build on `main` for a feature `main` does not have. What must never be
# silent is the other direction -- GPTQ sources present but ops missing
# means the patches stopped applying -- so that case still fails hard.
ARG AMDGPU_TARGETS
RUN python3 - "$AMDGPU_TARGETS" "$SGLANG_SRC" <<'PY'
import glob, os, re, subprocess, sys

target = sys.argv[1] if len(sys.argv) > 1 else "gfx1100"
sglang_src = sys.argv[2] if len(sys.argv) > 2 else "/opt/sglang-src"

import sgl_kernel
pkg = os.path.dirname(sgl_kernel.__file__)
sos = glob.glob(os.path.join(pkg, "common_ops*.so"))
if not sos:
    sys.exit(f"no common_ops*.so in {pkg} -- the kernel did not install")

# `strings` is not the real check -- a stray literal would satisfy it. Ask
# the object what code it actually carries: every offload bundle entry must
# be for the target, so a kernel built for the builder's own card (see the
# env_target_precedence patch) fails here instead of shipping.
#
# Two traps here, both found the hard way:
#
#  - llvm-objdump ships inside the ROCm tree, not on PATH.
#  - `--offloading` does not merely report on a HIP fat binary, it *extracts*
#    each bundle into a sidecar file next to the input. Run it in place and
#    site-packages/sgl_kernel fills up with common_ops.so.N.hipv4-... files,
#    which sgl_kernel's own loader then globs and chokes on -- the check
#    breaks the image it is checking. So inspect a copy in a temp dir.
import shutil, tempfile
objdump = (shutil.which("llvm-objdump")
           or next((p for p in glob.glob("/opt/rocm*/llvm/bin/llvm-objdump")
                    if os.access(p, os.X_OK)), None))
found = []
if objdump:
    with tempfile.TemporaryDirectory() as tmp:
        probe = os.path.join(tmp, os.path.basename(sos[0]))
        shutil.copy2(sos[0], probe)
        out = subprocess.run([objdump, "--offloading", probe],
                             capture_output=True, text=True).stdout
    found = sorted(set(re.findall(r"amdhsa--(gfx\d+)", out)))
if not found:  # no llvm-objdump, or a build without offload bundles
    out = subprocess.run(["strings", sos[0]], capture_output=True, text=True).stdout
    found = sorted(set(re.findall(r"gfx\d+", out)))
print("kernel targets:", ", ".join(found) or "(none)")
if found != [target]:
    sys.exit(f"kernel carries code for {found}, expected exactly ['{target}']")

gptq_src = os.path.join(sglang_src, "python", "sglang", "kernels", "aot",
                        "csrc", "gemm", "gptq", "gptq_kernel.cu")
if not os.path.exists(gptq_src):
    print(f"sgl_kernel OK: built for {target} (no AoT GPTQ tree in this sglang)")
    raise SystemExit(0)

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
# multimodal_gen is named separately from sglang because it is the diffusion
# stack's entry point and it pulls in a different slice of the dependency
# tree; an incomplete venv has shown up here while plain `import sglang`
# still worked.
#
# `sglang serve --help` is deliberately NOT here, tempting as it is. It
# reaches hardware while merely printing help (the diffusion path calls
# torch.cuda.get_device_properties(0) to decide what to print), so on a
# GPU-less builder it dies with "No HIP GPUs are available" no matter how
# healthy the image is. `sglang-rdna3 doctor` covers it on real hardware,
# which is where it belongs.
RUN python3 -c "import sglang" \
 && python3 -c "import sglang.multimodal_gen" \
 && python3 -c "import sglang.cli.main" \
 && sglang --help > /dev/null \
 && echo "sglang OK: LLM and diffusion stacks both import, CLI parses args"

WORKDIR /workspace
VOLUME ["/workspace", "/models"]
EXPOSE 30000

CMD ["sglang-rdna3", "doctor"]
