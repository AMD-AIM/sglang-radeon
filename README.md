# sglang-radeon-rdna3

Run [SGLang](https://github.com/sgl-project/sglang) on AMD Radeon **RDNA3**
GPUs (gfx1100 / gfx1101 / gfx1102 — Radeon PRO W7900, W7800, RX 7900 XTX).

SGLang officially supports only CDNA Instinct parts (gfx942 / gfx950) plus
gfx1250. RDNA3 is absent from the docs and is rejected outright by the kernel
build. ROCm itself supports these cards perfectly well — the gap is entirely in
SGLang's architecture allowlist and in a handful of platform assumptions.

This package closes that gap **from the outside**: it is an ordinary pip
package that patches nothing in the SGLang tree, so you can upgrade SGLang
independently.

Validated on 2× Radeon PRO W7900 (gfx1100, 48 GB each), ROCm 7.2.1,
PyTorch 2.9.1, SGLang `51b27f7`, serving **Qwen3.8-27B** (BF16, TP2) including
its vision path.

## Quick start

```bash
# 1. Install the plugin.
pip install --no-deps sglang-radeon-rdna3

# 2. Build sgl-kernel for your GPU (SGLang's HIP path hard-requires it and the
#    PyPI wheel is CUDA-only). Takes a couple of minutes.
git clone --depth 1 https://github.com/sgl-project/sglang.git
sglang-rdna3 build-kernel --sglang-src ./sglang --install

# 3. Check everything.
sglang-rdna3 doctor

# 4. Serve.
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --tp-size 2 \
  --attention-backend triton \
  --disable-cuda-graph \
  --dtype bfloat16 \
  --mem-fraction-static 0.85
```

`sglang-rdna3 serve-args <model>` prints a launch line tuned to the GPUs it
finds.

## What `doctor` tells you

```
  [ok  ] GPU 0: AMD Radeon Graphics gfx1100 48.0 GiB, 48 CUs
  [ok  ] GPU 1: AMD Radeon Graphics gfx1100 48.0 GiB, 48 CUs

  [ok  ] sgl_kernel importable
  [ok  ] sglang detects ROCm/HIP
  [ok  ] 22 backends, triton available
  [ok  ] 249 model architectures registered

  compat shims:
    - aiter_gemm_stubs: applied (3 stub modules)
    - fused_add_rms_norm_arity: applied
```

"249 model architectures registered" is the number that matters. If you see a
number under ten, `sgl_kernel` is missing or is a CUDA build, and almost
nothing will work.

## What this package actually fixes

**The kernel build refuses to run.** `setup_rocm.py` calls `sys.exit(1)` for
any target outside `gfx942 / gfx950 / gfx1250`. `build-kernel` patches the
allowlist — and also fixes something less obvious: upstream hands every
non-gfx942 target a 128 KB dynamic shared-memory budget, but RDNA3 has 64 KB
of LDS per workgroup, the same as gfx942, so it needs the 48 KB budget.
Patching only the allowlist gets you a kernel that misbehaves. The original
file is saved as `setup_rocm.py.orig` and the patch is idempotent.

**`import sglang` dies before any model registers.** SGLang's quark
quantization module imports `aiter.ops.triton.gemm.fused.*` under a bare
`if _is_hip:`, but shipping aiter builds put those kernels elsewhere. That
import sits on the path of every model module. These are CDNA-only MXFP4
GEMMs that RDNA3 cannot use, so we stub them and raise only if something
actually calls them.

**The first forward pass raises a `TypeError`.** `layers/layernorm.py` calls
`vllm._custom_ops.fused_add_rms_norm` with the legacy six-argument
out-of-place signature, but vLLM ≥ 0.16 ships a four-argument in-place kernel.
Any Gemma-style RMSNorm model — Qwen3.8-27B included — dies on its first
token. We wrap the kernel so both call shapes work. (This one is not
RDNA3-specific; it just happens to block us too.)

## Why a `.pth` file and not only the plugin entry point

SGLang has an entry-point plugin system, and this package registers with it.
But `load_plugins()` runs *after* `import sglang`, and the aiter failure above
happens *during* module import — so the entry point alone is structurally too
late. Installation therefore also drops a `.pth` file into site-packages,
which the `site` module executes at interpreter startup. It installs a
lightweight import hook that fires on the first `sglang*` import and then
removes itself, so processes that never touch SGLang pay nothing.

Set `SGLANG_RDNA3_ENABLE=0` to load the package but skip all patching — useful
for checking whether a shim is still needed after an SGLang upgrade.

## Known limitations on RDNA3

**No FP8.** RDNA3 has no native FP8, and SGLang correctly declines to enable
its fnuz paths (`is_fp8_fnuz()` tests for `gfx94`). Serve BF16/FP16, or a
weight-only quantization. A 27B model in BF16 needs ~55 GB and so wants TP2 on
48 GB cards.

**No CUDA graphs.** HIP graph capture hangs on gfx1100
([#30245](https://github.com/sgl-project/sglang/issues/30245)). Always pass
`--disable-cuda-graph`. This costs some latency at small batch sizes.

**MoE models do not work.** Fused-MoE hits an invalid memory access from a
codegen bug in Triton's AMD backend
([triton#10808](https://github.com/triton-lang/triton/issues/10808)). The
server starts, loads weights, then faults on the first MoE forward. Nothing
in SGLang or this package can fix it; it needs the upstream Triton fix. Dense
models are unaffected.

**Use the Triton attention backend.** `fa3`, `flashinfer`, `aiter` and friends
are CUDA- or CDNA-only.

**Performance is untuned.** SGLang ships no AMD MoE tuning configs at all (not
even for MI300X), and none of the hipBLASLt or Triton autotuning has been done
for gfx1100. Expect correctness, not peak throughput. Measured on 2× W7900,
Qwen3.8-27B BF16 TP2, 512-token prompts at concurrency 4: 32 tok/s output,
1.2 s TTFT, 92 ms per output token.

## Installing SGLang itself on ROCm

This is the part that bites people, and it is why this package declares **no
dependencies at all**.

SGLang's core dependency list hard-codes CUDA packages with no platform
markers — `cuda-python`, `cuda-tile`, `flashinfer_python[cu13]`,
`nvidia-cutlass-dsl[cu13]`. Letting pip resolve them will pull a CUDA build of
PyTorch **over your working ROCm one**. `humming-kernels[cu13]` will do it
through a transitive dependency even if you filter the obvious names.

Install with `--no-deps` and supply the rest yourself:

```bash
pip install --no-deps --no-build-isolation -e ./sglang/python
```

then add the non-CUDA runtime dependencies (also `--no-deps`). Pinning torch
with `-c constraints.txt` does not work — pip reports `ResolutionImpossible`.
See [docs/installation.md](docs/installation.md) for a working recipe.

## License

Apache-2.0.
