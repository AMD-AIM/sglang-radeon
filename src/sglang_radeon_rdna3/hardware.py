"""RDNA3 hardware detection and recommended serving defaults."""

from __future__ import annotations

from dataclasses import dataclass

#: gfx targets this package supports. gfx1100 (W7900/W7800/7900 XTX) is the
#: one we test against; 1101/1102 are the smaller Navi 32/33 parts and share
#: the same constraints.
RDNA3_TARGETS = ("gfx1100", "gfx1101", "gfx1102")


@dataclass(frozen=True)
class GpuInfo:
    index: int
    gfx: str
    name: str
    total_memory_gib: float
    compute_units: int

    @property
    def is_rdna3(self) -> bool:
        return self.gfx in RDNA3_TARGETS


def detect_gpus() -> list[GpuInfo]:
    """Return info for every visible GPU. Empty list if torch sees none."""
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []

    out: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        # gcnArchName looks like "gfx1100" or "gfx942:sramecc+:xnack-"
        gfx = getattr(p, "gcnArchName", "") or ""
        out.append(
            GpuInfo(
                index=i,
                gfx=gfx.split(":")[0],
                name=p.name,
                total_memory_gib=round(p.total_memory / 2**30, 1),
                compute_units=p.multi_processor_count,
            )
        )
    return out


def recommended_server_args(model_size_gib: float, gpus: list[GpuInfo]) -> dict:
    """Serving flags that are known-good on RDNA3.

    Rationale for each:
      attention_backend=triton  -- fa3/flashinfer/aiter backends are CDNA- or
          CUDA-only. The Triton backend has no gfx1100 exclusion and is the
          path validated in this project.
      disable_cuda_graph=True   -- HIP graph capture hangs on gfx1100
          (sgl-project/sglang#30245).
      dtype=bfloat16            -- RDNA3 has no native FP8. sglang correctly
          declines to enable fnuz paths (is_fp8_fnuz() checks for "gfx94"),
          so weights must stay 16-bit unless a weight-only quant is used.
      tp_size                   -- smallest power of two whose aggregate VRAM
          holds the weights with ~25% headroom for KV cache and activations.
    """
    usable = [g for g in gpus if g.is_rdna3] or gpus
    per_gpu = usable[0].total_memory_gib if usable else 48.0

    tp = 1
    while tp < len(usable) and model_size_gib > per_gpu * tp * 0.75:
        tp *= 2

    return {
        "attention_backend": "triton",
        "disable_cuda_graph": True,
        "dtype": "bfloat16",
        "tp_size": tp,
        "mem_fraction_static": 0.85,
    }
