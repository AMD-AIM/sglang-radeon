# Serving Qwen3.8-27B on Radeon PRO W7900

End-to-end walkthrough for the model this project was validated against.
Everything below was run on 2× W7900 (gfx1100, 48 GB each), ROCm 7.2.1.

## About the model

Qwen3.8-27B is a dense **vision-language** model with a hybrid attention
stack: 48 Gated DeltaNet (linear attention) layers interleaved with 16 full
attention layers, 64 in total, plus an in-checkpoint MTP head. Architecture
string is `Qwen3_5ForConditionalGeneration`. BF16 weights are 54 GB across 18
shards.

Two properties make it a good fit for RDNA3:

* **It is dense.** Fused-MoE is broken on gfx1100 by a Triton codegen bug, so
  MoE models are off the table for now. This model sidesteps that entirely.
* **Its linear attention is pure Triton.** SGLang's Gated DeltaNet backend
  selects a Triton `causal_conv1d` by default and only swaps in a CUDA version
  under `if is_cuda()`. The `fla/` kernels carry no CUDA gating at all. They
  compile and run correctly on gfx1100 — this was the main open question going
  in, and the answer is yes.

## Download

```bash
export HF_ENDPOINT=https://hf-mirror.com   # if huggingface.co is unreachable
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.8-27B --local-dir ./Qwen3.8-27B
```

About three minutes over the mirror.

## Serve (BF16, two GPUs)

54 GB of weights does not fit in one 48 GB card, so tensor-parallel across two:

```bash
python -m sglang.launch_server \
  --model-path ./Qwen3.8-27B \
  --tp-size 2 \
  --attention-backend triton \
  --disable-cuda-graph \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

Every flag matters:

| Flag | Why |
|---|---|
| `--tp-size 2` | 54 GB of weights across 2× 48 GB |
| `--attention-backend triton` | `fa3`/`flashinfer`/`aiter` are CUDA- or CDNA-only |
| `--disable-cuda-graph` | HIP graph capture hangs on gfx1100 (sglang#30245) |
| `--dtype bfloat16` | RDNA3 has no native FP8 |
| `--mem-fraction-static 0.85` | leaves room for the vision tower |

Expected startup: weights load in ~16 s, each GPU sits at 25.7 GB used with
21.9 GB free, and the server is ready in a little over a minute.

You will see `-mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc`
during startup. It is harmless — aiter tries to JIT a CDNA-only all-reduce
kernel, fails, and SGLang uses NCCL instead.

## Test it

Text:

```bash
curl -s localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is 17 times 23? Reply with just the number."}],"max_tokens":128,"temperature":0}'
```

The reply contains reasoning before the answer — Qwen3.8 runs in thinking mode
by default, emitting a `</think>` marker before the final text. Set
`reasoning_effort` to tune it; the default (`xhigh`) over-thinks short prompts.

Vision — pass an image as a data URL:

```bash
python - <<'EOF'
import base64, io, json, urllib.request
from PIL import Image, ImageDraw
im = Image.new("RGB", (200, 120), "white"); d = ImageDraw.Draw(im)
d.ellipse([20, 20, 100, 100], fill="red"); d.rectangle([120, 30, 180, 90], fill="blue")
buf = io.BytesIO(); im.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
req = urllib.request.Request(
    "http://localhost:30000/v1/chat/completions",
    data=json.dumps({"model": "x", "max_tokens": 200, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What shapes and colors do you see?"},
            {"type": "image_url", "image_url": {"url": url}}]}]}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
EOF
```

Correctly answers "a red circle and a blue square", confirming the vision
encoder works on gfx1100.

## Measured performance

`sglang.bench_serving`, random dataset, 512-token prompts, 128-token outputs,
16 requests at concurrency 4, 2× W7900 BF16 TP2:

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Output token throughput | 32.4 tok/s |
| Total token throughput | 192.1 tok/s |
| Mean TTFT | 1226 ms |
| Mean TPOT | 92.3 ms |

Reproduce with:

```bash
python -m sglang.bench_serving --backend sglang-oai \
  --host 127.0.0.1 --port 30000 --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 16 --max-concurrency 4
```

Treat these as a correctness-era baseline, not a performance claim. No
hipBLASLt or Triton autotuning has been done for gfx1100, and CUDA graphs —
worth a lot at low batch sizes — are unavailable.

## Single GPU with 4-bit weights (recommended)

A 4-bit GPTQ checkpoint fits one 48 GB card with room to spare — and is
**faster than two cards in BF16**, because it drops the tensor-parallel
communication entirely.

| | 2x W7900 BF16 | 1x W7900 GPTQ INT4 |
|---|---|---|
| VRAM used | 25.7 GB per GPU | **19.0 GB total** |
| Output throughput | 32.4 tok/s | **47.3 tok/s** |
| Mean TPOT | 92.3 ms | **54.3 ms** |
| Mean TTFT | 1226 ms | 1161 ms |
| Weight load | 15.7 s | 11.4 s |

Same benchmark on both: 512-token prompts, 128-token outputs, 16 requests at
concurrency 4.

### Getting a checkpoint that actually works

This is the part that wastes people's time. **Check `quantization_config` in
`config.json` before downloading 20 GB**: you need `"quant_method": "gptq"`
(or `"awq"`). Many repositories named "...-AWQ" or "...-INT4" are actually
**compressed-tensors** `pack-quantized`, which routes through
`gptq_marlin_repack` — a CUDA-only kernel that SGLang imports under
`if _is_cuda:` but calls unconditionally, so on ROCm you get:

```
NameError: name 'gptq_marlin_repack' is not defined
```

Verified working: `palmfuture/Qwen3.8-27B-GPTQ-Int4` (4-bit, group size 32).

```bash
hf download palmfuture/Qwen3.8-27B-GPTQ-Int4 --local-dir ./Qwen3.8-27B-GPTQ
```

### Serving it

```bash
SGLANG_MAMBA_CONV_DTYPE=float16 \
python -m sglang.launch_server \
  --model-path ./Qwen3.8-27B-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --tp-size 1 --attention-backend triton --disable-cuda-graph \
  --mem-fraction-static 0.85 --port 30000
```

Three settings are non-obvious and all three are required:

* **`--quantization gptq`** — without it SGLang auto-selects `gptq_marlin` and
  stops with `gptq_marlin quantization is currently not supported in ROCm`.
  Plain `gptq` is a separate, portable method.
* **`--dtype float16`** — GPTQ rejects bfloat16 outright.
* **`SGLANG_MAMBA_CONV_DTYPE=float16`** — the least obvious one. Qwen3.8's
  Gated DeltaNet convolution state cache has its own dtype, independent of
  `--dtype`, defaulting to bfloat16. Leave it and the server starts happily,
  then dies on the first request:

  ```
  RuntimeError: Index put requires the source and destination dtypes match,
  got BFloat16 for the destination and Half for the source
  ```

### This needs the patched kernel

Upstream never compiles the GPTQ kernel for ROCm — `gptq_kernel.cu` is absent
from `setup_rocm.py`'s source list, and `gptq_shuffle`/`gptq_gemm` are
registered only in the CUDA extension. You would otherwise hit:

```
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'gptq_shuffle'
```

`sglang-rdna3 build-kernel` adds the source, registers the ops, and repairs
two stale `#ifdef USE_ROCM` branches inside the kernel that assumed a
pre-ROCm-7 `half2` layout. Pass `--no-gptq` to skip all of that and build base
RDNA3 support only.

Note this is weight-only quantization — activations stay 16-bit, which is what
you want on RDNA3 since there is no native FP8. Avoid the `quark`/MXFP4 paths;
they need CDNA-only aiter kernels and this plugin stubs them out precisely
because they cannot work here.
