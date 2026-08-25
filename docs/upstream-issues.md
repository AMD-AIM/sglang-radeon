# Upstream issues found while enabling RDNA3

Every workaround in this package corresponds to a defect that could be fixed
in SGLang itself. They are listed here so the list can shrink over time, and
so anyone filing PRs has the details to hand.

Reference environment: 2-4x Radeon PRO W7900 (gfx1100, 48 GB), ROCm 7.2.1,
PyTorch 2.9.1+rocm, vLLM 0.16.1.dev0+rocm721, SGLang at `51b27f7`.

---

## 1. The kernel build rejects RDNA3 outright

`python/sglang/kernels/aot/setup_rocm.py`

```python
if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:
    print(f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'...")
    sys.exit(1)
```

ROCm officially supports gfx1100 (W7900, W7800, RX 7900 XTX are all in AMD's
compatibility matrix), so this is a policy gate rather than a hardware limit.
Adding `gfx1100/1101/1102` is sufficient to build, and the result runs.

Related: [#9417](https://github.com/sgl-project/sglang/issues/9417),
[#27519](https://github.com/sgl-project/sglang/issues/27519).

**Fix:** widen the allowlist.

---

## 2. RDNA3 gets the wrong shared-memory budget

Same file, a few lines down:

```python
topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target == "gfx942" else 32 * 1024 * 4
```

RDNA3 has **64 KB of LDS per workgroup — the same as gfx942** — so it belongs
in the 48 KB branch, not the 128 KB one. Community reports about patching the
allowlist do not mention this, which means anyone following them builds a
kernel configured for shared memory its hardware does not have.

**Fix:** select on a set of 64 KB-LDS architectures rather than on gfx942
alone.

---

## 3. `import sglang` dies on ROCm when aiter is installed

`srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py:32`

```python
if _is_hip:
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...
```

Shipping aiter builds place these kernels at `aiter/ops/triton/`, with no
`gemm/fused` subpackage, so the import raises `ModuleNotFoundError`. It is
reached from `layers/quantization/__init__.py:57`, which is on the import path
of every model module — so the failure is not local to quark. Model
registration collapses from ~249 architectures to 4, and the error message
says nothing about aiter.

These are MXFP4 GEMMs for CDNA. RDNA3 has no FP4/FP8 hardware and can never
use them.

**Fix:** guard the import with try/except and degrade, rather than assuming
every HIP device is CDNA with a matching aiter version.

---

## 4. `fused_add_rms_norm` is called with the wrong arity

`srt/layers/layernorm.py:1158`

```python
# vllm API: fused_add_rms_norm(out, input, residual_out, residual, weight, eps)
fused_add_rms_norm(out, x, residual_out, residual, w, self.variance_epsilon)
```

vLLM >= 0.16 ships a four-argument in-place kernel:

```python
fused_add_rms_norm(input, residual, weight, epsilon) -> None
```

So the first forward pass of any Gemma-style RMSNorm model raises
`TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were
given`. Qwen3.8-27B hits this immediately.

Not RDNA3-specific — it affects any ROCm deployment on current vLLM. The same
mismatch exists on the aiter path at line 114, which aliases
`rmsnorm2d_fwd_with_add` (also four arguments) and sets
`_has_vllm_rms_norm = True`.

**Fix:** detect the arity, or pin the expected vLLM API.

---

## 5. GPTQ is never compiled for ROCm

Two gaps, both in `python/sglang/kernels/aot/`:

* `csrc/gemm/gptq/gptq_kernel.cu` is missing from `setup_rocm.py`'s source
  list.
* `gptq_shuffle` and `gptq_gemm` are registered only in
  `csrc/common_extension.cc` (CUDA); `csrc/common_extension_rocm.cc` omits
  them.

Result: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
'gptq_shuffle'`.

This matters a lot on 48 GB cards. With GPTQ working, a 27B model runs on a
single GPU at 19 GB and is **46% faster** than the same model split BF16
across two (47.3 vs 32.4 tok/s), since it avoids tensor-parallel
communication.

**Fix:** add the source file and register both ops in the ROCm extension.

---

## 6. Stale `USE_ROCM` branches in the GPTQ kernel

`csrc/gemm/gptq/gptq_kernel.cu` already carries ROCm paths, written when HIP
exposed `half2` lanes as `unsigned short`:

```cpp
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
```

In ROCm 7.x those lanes are `__half`, matching CUDA, so the ROCm branch no
longer compiles:

```
error: no matching function for call to '__ushort_as_half'
note: no known conversion from '__half' to 'unsigned short'
```

Someone did port this once; the port simply predates the HIP change.

Two sites, on the zero-init and accumulate paths. Note `res2 = {}` cannot be
used directly on HIP either — `__half2` has several `operator=` overloads and
brace-init is ambiguous — so the zero must be built explicitly with
`__halves2half2(__float2half(0.f), __float2half(0.f))`.

**Fix:** let modern HIP take the CUDA path.

---

## 7. GPTQ and hybrid linear attention disagree about dtype

GPTQ requires float16. Qwen3.8's Gated DeltaNet convolution state cache takes
its dtype from `SGLANG_MAMBA_CONV_DTYPE` (default bfloat16,
`configs/mamba_utils.py:47`), independent of `--dtype`. The server starts
normally and then fails on the first request:

```
RuntimeError: Index put requires the source and destination dtypes match,
got BFloat16 for the destination and Half for the source
```

at `layers/attention/linear/gdn_backend.py:614`.

Workaround: set `SGLANG_MAMBA_CONV_DTYPE=float16` explicitly.

**Fix:** default the conv state dtype to the model's compute dtype instead of
an unconditional bfloat16.

---

## 8. `compressed-tensors` 4-bit is unusable on ROCm

`srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
imports `gptq_marlin_repack` under `if _is_cuda:` (line 46) but calls it
unconditionally (line 250), giving `NameError: name 'gptq_marlin_repack' is
not defined` on ROCm.

This is worth fixing because most 4-bit checkpoints on the Hub — including
many labelled "AWQ" — are compressed-tensors `pack-quantized`, so ROCm users
find that almost nothing they download works, with an error that does not
explain why.

**Fix:** raise a clear "not supported on ROCm, use quant_method=gptq/awq"
error at load time, or provide a non-Marlin repack path.

---

## Packaging problems (not defects, but they cost time)

**CUDA packages are unconditional core dependencies.** `cuda-python`,
`cuda-tile`, `flashinfer_python[cu13]`, `nvidia-cutlass-dsl[cu13]` and
`nvidia-mathdx` carry no platform markers, so `pip install -e .` on a ROCm box
resolves them and installs a **CUDA build of PyTorch over the ROCm one**.
`humming-kernels[cu13]` does it transitively even if the obvious names are
filtered. Platform markers would fix this.

**`sglang-kernel` is a core dependency too**, and the PyPI wheel is CUDA-only.
Any routine `pip install -r` silently replaces a locally built ROCm kernel,
and the resulting error (`libnvrtc.so.13: cannot open shared object file`)
does not point at the cause.

**`kernels_data` is imported by the build but never declared** as a build
requirement.
