# MiniMax-H3 on Radeon RDNA3

MiniMax-H3 is **not** a text LLM. It is a 33 B omni-modal transformer that
generates video *and* synchronized audio from a single packed multimodal
sequence. It is served by **SGLang-Diffusion**
(`sglang.multimodal_gen`), a separate subsystem from the `sglang.srt` LLM
engine, so almost none of the LLM-side guidance transfers.

This page records what is required to run it on RDNA3 and what remains
unresolved. Read it before committing hardware to the attempt.

## Reality check

Upstream scopes SGLang-Diffusion's AMD support to **Instinct only**:

> AMD GPUs (MI300X, MI325X, MI355X) […] On AMD platforms, we use the Triton
> attention backend and leverage AITER kernels for optimized layernorm.

AITER is CDNA-only. On the LLM side that is a non-issue (AITER defaults off),
but here it is named as part of the supported path. The situation is better
than that sentence implies, though — inspecting the source:

* AITER imports in `runtime/layers/layernorm.py` are guarded by `USE_AITER`,
  which is off by default.
* The hard `import aiter` calls live in models H3 does not use
  (`wanvideo.py`, `lingbot_world.py`) and in optional backends
  (`backends/aiter.py`, `backends/aiter_sage.py`).
* `runtime/platforms/rocm.py` contains **no** gfx allowlist.
* A `backends/sdpa.py` attention backend exists as a portable fallback.

So there is no structural reason H3 cannot run on gfx1100. What there is
instead is a lot of untested surface.

Also note the plugin mechanism does not reach here: SGLang's platform plugin
system covers `SRTPlatform`; the multimodal `MMPlatform` is marked as future
work. Shims for the diffusion stack cannot be delivered the same way.

## Layout: point at a partition, not the repo root

This is the first thing that goes wrong. The repository root is **not** a
loadable pipeline. `MiniMaxH3ReleaseMetadata.from_model_index` requires a
`_minimax_h3` key that only the partition subdirectories carry:

```
MiniMax-H3/
├── model_index.json         <- no _minimax_h3; loading this fails
├── transformer/  text_encoder/  vae/   <- shared component weights
├── FL2VA/                   <- a real pipeline (partition "fl2va")
│   ├── model_index.json     <- has _minimax_h3, tasks ["t2va","fl2va"]
│   └── transformer/ text_encoder/ video_vae/ audio_vae/ ...
└── Ref2VA/                  <- a real pipeline (partition "ref2va")
```

Pointing at the root gives:

```
ValueError: model_index.json._minimax_h3 must be an object
```

Choose the partition by task: **FL2VA** for text-to-video and
first/last-frame conditioning (`t2va`, `fl2va`); **Ref2VA** for reference
images, video clips and audio.

Each partition is a self-contained ~134 GB checkpoint. The full repository is
464 GB; do not fetch all of it.

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

## Memory

The FL2VA partition is roughly 62 GB of DiT transformer, 62 GB of Qwen3-VL
text encoder and 10 GB of VAEs. In BF16 that does not fit on two 48 GB cards;
plan on **four**. There is no usable shortcut: the memory-saving GGUF path
upstream documents is annotated CUDA-only and `--tp-size 1`, and RDNA3 has no
FP8 to fall back on.

## Running it

```python
import os
os.environ["SGLANG_USE_AITER"] = "0"                # CDNA-only kernels
os.environ["SGLANG_DIFF_ATTENTION_BACKEND"] = "sdpa"  # portable fallback

def main():
    from sglang.multimodal_gen import DiffGenerator
    gen = DiffGenerator.from_pretrained(
        model_path="./MiniMax-H3/FL2VA",   # a partition, never the root
        num_gpus=4,
    )
    gen.generate(sampling_params_kwargs=dict(
        prompt="A red balloon drifting over a calm lake at sunrise.",
        output_path="./h3_out/",
        save_output=True,
    ))

if __name__ == "__main__":   # required: workers are spawned, not forked
    main()
```

The `if __name__ == "__main__"` guard is not optional. Without it every worker
re-executes the module and you get a wall of

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

which obscures the real error.

Install the diffusion extras first, with the same `--no-deps` discipline used
for SGLang itself (`nvidia-modelopt`, `st_attn` and `vsa` are CUDA-only and
must be dropped) — see [installation.md](installation.md).

## Status

Distributed init and RCCL bring-up succeed on 4× W7900; `import
sglang.multimodal_gen` is clean once the CUDA-only extras are excluded. Full
generation is **not yet verified end to end** on RDNA3. Treat this page as a
working map of the terrain rather than a supported configuration, and expect
to debug.

If you only need a text LLM, use Qwen3.8-27B — that path *is*
verified. See [qwen3.8-27b.md](qwen3.8-27b.md).
