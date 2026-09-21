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

### Not installing aiter does not avoid it, and the surface is wide

The obvious workaround — leave aiter out of a ROCm image that cannot use it —
does not work, because these imports are unconditional under `if _is_hip:`.
On `v0.5.19`, `import sglang.multimodal_gen` fails with
`ModuleNotFoundError: No module named 'aiter'` **even with a GPU present**.

Nor is it one import to stub. Feeding SGLang a synthetic `aiter` package and
following the failures gives a chain, each hop only visible once the previous
one is satisfied:

1. `quark/schemes/quark_w4a4_mxfp4.py` — `aiter.ops.triton.gemm.fused...`
2. `diffusers/utils/import_utils.py` — probes with
   `importlib.util.find_spec("aiter")`, so a stub without a real
   `__spec__` raises `ValueError: aiter.__spec__ is None`. A stub complete
   enough to get past this also tells diffusers aiter *is* available, which
   is worse than the original failure.
3. `quark_int4fp8_moe.py:30` — `from aiter.ops.shuffle import shuffle_weight`

…and it continues (`aiter.utility.fp4_utils`, `aiter.ops.triton.quant`,
`aiter.ops.flydsl.moe_common`). Each stub moves the failure rather than
fixing it.

This package therefore does **not** ship an aiter shim. Faking a hardware
acceleration library well enough to satisfy third-party capability probes is
not a fix; it is a way to get CDNA code paths selected on a card that cannot
run them. The real fix is the one above — make the imports lazy or guarded —
and it has to happen upstream.

`main` has since made the diffusion path tolerant (its `mxfp4.py` uses
try/except), which is why this package defaults to `main` rather than
`v0.5.19`.

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

---

## 9. Inline PTX in the diffusion Triton kernels

`kernels/ops/diffusion/common/numerics.py` and
`kernels/ops/diffusion/norm/layernorm_modulate_triton.py` embed PTX assembly
via `tl.inline_asm_elementwise` — `mul.rn.f32`, `div.rn.f32`,
`rsqrt.approx.f32`, `rcp.approx.f32`. PTX is NVIDIA-only, so the AMD backend
fails outright:

```
error: couldn't allocate output register for constraint 'f'
```

The intent is bit-exact agreement with CUDA output, which is a reasonable
goal — but it makes these kernels unbuildable on every non-NVIDIA backend, and
the error surfaces during VAE decode rather than at compile time, which sends
you looking in the wrong place entirely.

**Fix:** guard the PTX behind a platform check and fall back to native Triton
(`x * y`, `x / y`, `tl.rsqrt`, `1.0 / x`) elsewhere. Accuracy differs by at
most an ulp.

---

## 10. The MiniMax-H3 denoise loop has a CUDA/MPS allowlist

`multimodal_gen/.../minimax_h3/stages/denoising.py:623`:

```python
if not (current_platform.is_cuda() or current_platform.is_mps()):
    raise RuntimeError("MiniMax H3 full-loop denoise requires CUDA or MPS")
```

Everything after this gate is device-agnostic PyTorch, and
`get_local_torch_device()` already returns `cuda:0` on ROCm since HIP reuses
the CUDA device namespace. With the gate widened, the loop runs correctly on
gfx1100 and produces valid H.264 + AAC output.

**Fix:** include ROCm, or test `is_cuda_alike()`, which already treats ROCm as
CUDA-like elsewhere in this codebase.

---

## 11. `_apply_qk_norm` bypasses its own fallback on ROCm

`multimodal_gen/runtime/models/dits/minimax_h3.py:381`:

```python
if (q.is_cuda and q.dtype == _BF16_DTYPE and ...):
    fused_inplace_qknorm(q, k, ...)   # CUDA-only JIT kernel
    return q, k
return q_norm(q), k_norm(k)           # unreachable on ROCm
```

`q.is_cuda` is True for HIP tensors, so ROCm takes the fused path and the JIT
build fails. The correct fallback is one line below.

**Fix:** test for a compilable kernel rather than for `is_cuda` — e.g. via the
`can_use_*` predicate the other call sites already use.

---

## 12. Component weights are only discovered one directory deep

`multimodal_gen/runtime/utils/hf_diffusers_utils.py:172`:

```python
def _has_local_weight_files(component_path):
    return any(glob.glob(os.path.join(component_path, pattern))
               for pattern in _WEIGHT_FILE_PATTERNS)
```

MiniMax-H3 ships its video VAE at `video_vae/source/model.safetensors`, one
level down, so a complete checkout is judged incomplete. SGLang then tries to
re-download it, passes the local filesystem path to the Hub API, and fails
with `HFValidationError: Repo id must be in the form 'repo_name' or
'namespace/repo_name'` — an error that says nothing about the real problem.

**Fix:** search recursively, or at least one extra level.

---

## 13. GPU autodetection silently overrides an explicit `AMDGPU_TARGET`

`python/sglang/kernels/aot/setup_rocm.py:67`

```python
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
```

The environment variable is read and then thrown away whenever any GPU is
visible, so **cross-compiling is impossible**. Asking for `gfx1100` on a
machine with an MI355X in it produces a `gfx950` kernel — with no warning,
a zero exit status, and a wheel that installs and imports cleanly. The
mismatch only surfaces on the target card, far from its cause.

It is easy to miss because CI never sees it: a GPU-less builder takes the
`else` branch and the variable survives. It bites the developer building on
their own workstation, which is also the person least likely to suspect the
build script.

Measured: `build-kernel --target gfx1100` on a gfx950 host, then

```
$ llvm-objdump --offloading common_ops.cpython-312-x86_64-linux-gnu.so
hipv4-amdgcn-amd-amdhsa--gfx950      # every bundle, no gfx1100 anywhere
```

**Fix:** treat an explicit request as authoritative and autodetect only when
none was made. Worth reordering rather than merely guarding, so the "torch.cuda
not available" message does not get printed on a machine whose GPU is present
and simply lost to an explicit request.

> Note for anyone verifying this themselves: `llvm-objdump --offloading`
> *extracts* the bundles into sidecar files next to its input rather than only
> reporting on them. Run it on a copy — in place, it litters
> `site-packages/sgl_kernel/` with files that sgl_kernel's own loader then
> globs and fails on.
