# MiniMax-H3 on Radeon RDNA3

MiniMax-H3 is **not** a text LLM. It is a 33B omni-modal transformer that
generates video *and* synchronized audio from one packed multimodal sequence,
served by **SGLang-Diffusion** (`sglang.multimodal_gen`) rather than the
`sglang.srt` LLM engine. Almost none of the LLM-side guidance transfers.

**Status: the denoising loop runs on gfx1100.** Every stage of the pipeline
executes and the GPU sits at 95% during denoising. It is, however, *slow* —
25-65 s per step and degrading, because the 62 GB DiT has to be streamed layer
by layer from host RAM. Treat this as "the blockers are broken" rather than
"this is a practical way to generate video today."

Upstream scopes SGLang-Diffusion's AMD support to Instinct only (MI300X /
MI325X / MI355X). Everything below is what it takes to get past that.

## Layout: point at a partition, not the repository root

The repository root is not a loadable pipeline.
`MiniMaxH3ReleaseMetadata.from_model_index` requires a `_minimax_h3` key that
only the partition subdirectories carry:

```
MiniMax-H3/
├── model_index.json         <- no _minimax_h3; loading this fails
├── FL2VA/                   <- a real pipeline (partition "fl2va")
│   ├── model_index.json     <- has _minimax_h3, tasks ["t2va","fl2va"]
│   └── transformer/ text_encoder/ video_vae/ audio_vae/ ...
└── Ref2VA/                  <- a real pipeline (partition "ref2va")
```

Pointing at the root gives `ValueError: model_index.json._minimax_h3 must be
an object`.

Pick by task: **FL2VA** for text-to-video and first/last-frame conditioning
(`t2va`, `fl2va`); **Ref2VA** for reference images, video and audio. Each
partition is a self-contained ~134 GB checkpoint — the full repository is
464 GB, so do not fetch all of it.

```bash
export HF_ENDPOINT=https://hf-mirror.com   # if huggingface.co is blocked
python - <<'EOF'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"       # the xet path errors out here
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("MiniMaxAI/MiniMax-H3", local_dir="./MiniMax-H3",
                  allow_patterns=["FL2VA/*"], max_workers=8)
EOF
```

Note the plugin also fixes a completeness check that would otherwise reject
this download: SGLang globs one directory level for weights, but H3 ships its
video VAE at `video_vae/source/model.safetensors`. Without the fix SGLang
judges a complete checkpoint incomplete, tries to "repair" it by passing your
local path to the Hub API, and dies with `HFValidationError: Repo id must be
in the form 'repo_name' or 'namespace/repo_name'`.

## Memory: one GPU, not four

This is counterintuitive and cost us several OOMs.

The FL2VA partition is ~62 GB of DiT, ~62 GB of Qwen3-VL text encoder and
~10 GB of VAEs. On 48 GB cards:

* **`num_gpus=4` does not split the DiT.** Each rank loads a full copy —
  46.2 GB per card — and then dies with `HIP out of memory. Tried to allocate
  294.00 MiB. GPU 0 has ... 106.00 MiB is free`. The parallelism here is
  sequence parallelism, not weight sharding.
* **CPU offload with 4 GPUs is worse.** The host-side copies are per-worker,
  so four ranks means four copies: host RAM climbed past 360 GB and the
  container was OOMKilled. (Check with
  `kubectl get pod X -o jsonpath='{.status.containerStatuses[0].lastState}'`
  — you will see `"reason":"OOMKilled"` rather than any Python traceback.)

What works is **one GPU with layer-wise offload**: a single CPU-side copy,
with only a fraction of the DiT resident on the GPU at a time. Budget roughly
200 GB of host RAM.

## Running it

```python
import os
os.environ["SGLANG_USE_AITER"] = "0"        # AITER kernels are CDNA-only
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")


def main():
    from sglang.multimodal_gen import DiffGenerator

    gen = DiffGenerator.from_pretrained(
        model_path="./MiniMax-H3/FL2VA",   # a partition, never the root
        num_gpus=1,
        layerwise_offload_components=["dit", "text_encoder", "vae"],
        dit_layerwise_resident_layers=0.2,
        attention_backend="torch_sdpa",
    )
    gen.generate(sampling_params_kwargs=dict(
        task="t2va",
        prompt="A red balloon drifting over a calm lake at sunrise.",
        target=dict(
            short_edge=768,          # the only verified value
            aspect_ratio="16:9",
            duration_seconds=4.0,    # valid range is 4.0-15.0
        ),
        output_path="./h3_out/",
        save_output=True,
    ))


if __name__ == "__main__":   # required: workers are spawned, not forked
    main()
```

Every argument here is load-bearing:

| Argument | Why |
|---|---|
| `model_path=.../FL2VA` | the root has no `_minimax_h3` metadata |
| `num_gpus=1` | more ranks means more full copies, not sharding |
| `layerwise_offload_components` | the 62 GB DiT does not fit 48 GB |
| `attention_backend="torch_sdpa"` | the default `"fa"` reaches AITER's `fmha_v3_varlen_fwd`, which fails with `invalid argument` on gfx1100. Note the name is `torch_sdpa`, not `sdpa`, and it is a constructor argument — not an environment variable |
| `task=` / `target=` | both are required; omitting either raises immediately |

The `if __name__ == "__main__"` guard is not optional. Without it every worker
re-executes the module and a wall of `RuntimeError: An attempt has been made
to start a new process before the current process has finished its
bootstrapping phase` buries the real error.

Install the diffusion extras with the same `--no-deps` discipline used for
SGLang itself — `nvidia-modelopt`, `st_attn` and `vsa` are CUDA-only and must
be dropped. See [installation.md](installation.md).

## What the plugin fixes for you

**The platform allowlist.** `minimax_h3/stages/denoising.py:623` reads

```python
if not (current_platform.is_cuda() or current_platform.is_mps()):
    raise RuntimeError("MiniMax H3 full-loop denoise requires CUDA or MPS")
```

Everything past that gate is device-agnostic PyTorch, and on ROCm
`get_local_torch_device()` already returns `cuda:0` because HIP reuses the
CUDA device namespace. The allowlist simply predates ROCm support.

**CUDA-only JIT kernels.** Thirty-four sources under `sglang/kernels/jit/csrc`
hard-include `cuda_bf16.h` / `cuda_fp16.h` or assume a 32-bit warp mask, so
they fail to compile on HIP:

```
fatal error: 'cuda_bf16.h' file not found
amd_warp_sync_functions.h:277: static assertion failed ...
    'sizeof(unsigned int) == 8': The mask must be a 64-bit integer
```

These are genuine source incompatibilities — porting them means rewriting for
wave64. Each is opt-in behind a `can_use_*` predicate with a PyTorch fallback,
so the plugin reports False for all 36 predicates. Set
`SGLANG_RDNA3_ALLOW_JIT=1` to leave them enabled and check whether any have
since been ported.

**One kernel that skips its own predicate.** `minimax_h3.py:381` gates on
`q.is_cuda` — which is True for HIP tensors — and calls
`fused_inplace_qknorm` directly:

```python
if (q.is_cuda and q.dtype == _BF16_DTYPE and ...):
    fused_inplace_qknorm(q, k, ...)
    return q, k
return q_norm(q), k_norm(k)      # the fallback, unreachable on ROCm
```

There is no try/except here, only a condition, so making the kernel raise does
not help. The plugin replaces `_apply_qk_norm` with its own fallback branch.
It does this *in the DiT module only*: patching the kernel globally breaks
text-encoder construction, which uses the same kernel and has no fallback
(`Failed to load customized text_encoder; native fallback is disabled`).

## If you only need a text model

Use Qwen3.8-27B — that path is fully verified and fast. See
[qwen3.8-27b.md](qwen3.8-27b.md).
