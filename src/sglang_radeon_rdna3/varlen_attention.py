"""A varlen attention primitive for RDNA3, built on PyTorch SDPA.

SGLang-Diffusion funnels *every* variable-length attention call through one
symbol::

    sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func

On RDNA3 that symbol is unusable. It dispatches on ``ver`` (default 3) to
FlashAttention-3, which is Hopper-only, and the ``flash_attn`` package in the
ROCm image is an empty namespace package -- ``flash_attn.__file__`` is
``None`` and ``from flash_attn import flash_attn_varlen_func`` raises
ImportError. There is no varlen kernel on this platform at all.

That single symbol is also the *only* thing standing between RDNA3 and
VDN-H3. ``hybrid_window_attn_h3.py`` -- which VDN-H3 hard-requires, raising
``ValueError`` if any other backend is selected -- does its windowing by
*gathering rows into varlen segments* and then making plain dense calls. It
never asks the kernel for a sliding window, never asks for causality, and
never asks for the log-sum-exp.

So the feature surface we actually have to reproduce is small, and a packed
varlen call decomposes exactly into per-segment dense attention. That is
expressible in SDPA with no approximation: this is a *slower* implementation,
not a *different* one.

## What this is not

It is not a FlashAttention port. Two costs are inherent to doing it this way,
and both are worth stating plainly rather than discovering later:

* **It syncs.** ``cu_seqlens`` lives on the GPU and we need the segment
  bounds on the host to slice. Real varlen kernels read them on-device. One
  ``.tolist()`` per call is unavoidable at the Python level.
* **It may materialise the score matrix.** Whether SDPA picks a fused
  backend on gfx1100 or falls back to the math backend decides whether peak
  memory is O(L) or O(L^2). We do not assume; see ``sdpa_backend_in_use``.

## Refusals are deliberate

Everything this module cannot reproduce exactly raises. Attention that is
silently wrong does not crash and does not look wrong -- it produces a video
that is merely worse, and nothing upstream of the pixels reports anything.
An exception at the first call is enormously cheaper than that, and the
callers VDN-H3 uses never pass these arguments anyway.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Set by :func:`flash_attn_varlen_func` the first time it runs, so
#: ``sglang-rdna3 doctor`` can report which SDPA backend actually served the
#: call rather than guessing from the platform.
SDPA_BACKEND_OBSERVED: str | None = None


class VarlenFeatureUnsupported(NotImplementedError):
    """A varlen argument this shim cannot reproduce exactly was passed.

    Raised instead of ignoring the argument. See the module docstring.
    """


def _reject_unsupported(
    *,
    window_size: tuple[int, int],
    softcap: float,
    return_softmax_lse: bool,
    q_descale: Any,
    k_descale: Any,
    v_descale: Any,
    sinks: Any,
    score_mod: Any,
    qv: Any,
    page_table: Any,
    seqused_q: Any,
    seqused_k: Any,
) -> None:
    """Refuse the arguments that would change the numbers.

    Grouped into one function so the rejection list is readable as a list --
    it is the honest statement of this shim's scope, and it should be easy to
    check against a caller.
    """
    bad: list[str] = []
    if window_size not in ((-1, -1), None):
        # The hybrid-window backend does its windowing by gathering rows, not
        # by asking the kernel -- so nothing on the VDN-H3 path needs this.
        bad.append(f"window_size={window_size!r} (sliding window)")
    if softcap:
        bad.append(f"softcap={softcap!r}")
    if return_softmax_lse:
        # SDPA does not return the log-sum-exp. Only ring/Ulysses attention
        # needs it to merge partial results, and VDN-H3 refuses ring
        # parallelism outright, so this is unreachable on the path we care
        # about. Synthesising a plausible-looking LSE would be much worse
        # than saying no.
        bad.append("return_softmax_lse=True")
    for name, val in (("q_descale", q_descale), ("k_descale", k_descale),
                      ("v_descale", v_descale)):
        if val is not None:
            bad.append(f"{name} (FP8 dequant scales)")
    if sinks is not None:
        bad.append("sinks (attention sinks)")
    if score_mod is not None:
        bad.append("score_mod")
    if qv is not None:
        bad.append("qv (fused QV)")
    if page_table is not None:
        bad.append("page_table (paged KV)")
    for name, val in (("seqused_q", seqused_q), ("seqused_k", seqused_k)):
        if val is not None:
            bad.append(f"{name}")
    if bad:
        raise VarlenFeatureUnsupported(
            "sglang-radeon's RDNA3 varlen attention does not implement: "
            + ", ".join(bad)
            + ". It is an exact SDPA decomposition of the dense, non-windowed "
            "varlen call that VDN-H3 uses; anything else would have to be "
            "approximated, and wrong attention is silent. See "
            "sglang_radeon_rdna3/varlen_attention.py."
        )


def _segment_lengths(cu_seqlens: torch.Tensor) -> list[int]:
    """Host-side segment lengths from a cumulative-offset tensor.

    This is the sync the module docstring warns about. Doing it once per call
    and reusing the list is the whole mitigation -- do not call this twice.
    """
    bounds: Sequence[int] = cu_seqlens.tolist()
    return [int(bounds[i + 1]) - int(bounds[i]) for i in range(len(bounds) - 1)]


def _expand_kv_heads(t: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Broadcast KV heads up to query heads for grouped-query attention.

    ``t`` is ``[tokens, kv_heads, head_dim]``. FlashAttention does this inside
    the kernel for free; here it costs a real copy. VDN-H3's DiT is
    multi-head (kv_heads == q_heads), so on the path we care about this is a
    no-op -- it exists so that a GQA model does not get silently wrong
    results from a shape mismatch that happens to broadcast.
    """
    kv_heads = t.shape[1]
    if kv_heads == num_q_heads:
        return t
    if num_q_heads % kv_heads != 0:
        raise VarlenFeatureUnsupported(
            f"query heads ({num_q_heads}) is not a multiple of kv heads "
            f"({kv_heads}); this is not a GQA layout"
        )
    return t.repeat_interleave(num_q_heads // kv_heads, dim=1)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
          scale: float | None, causal: bool) -> torch.Tensor:
    """SDPA on ``[batch, heads, seq, dim]``, returned in the same layout."""
    global SDPA_BACKEND_OBSERVED
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale,
    )
    if SDPA_BACKEND_OBSERVED is None:
        SDPA_BACKEND_OBSERVED = _probe_sdpa_backend()
    return out


def _probe_sdpa_backend() -> str:
    """Which SDPA backend torch will pick here -- reported, never assumed.

    This decides whether peak memory is O(L) or O(L^2), which on a 48 GB card
    is the difference between VDN-H3 running and VDN-H3 OOMing. gfx1100's
    support for the fused backends moves with the ROCm version, so reading it
    off the platform would be a guess.
    """
    try:
        from torch.backends.cuda import (
            flash_sdp_enabled,
            mem_efficient_sdp_enabled,
            math_sdp_enabled,
        )
    except Exception:  # pragma: no cover - torch too old to ask
        return "unknown"
    enabled = [
        name for name, fn in (
            ("flash", flash_sdp_enabled),
            ("mem_efficient", mem_efficient_sdp_enabled),
            ("math", math_sdp_enabled),
        )
        if fn()
    ]
    return "+".join(enabled) if enabled else "none"


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    seqused_q: Any = None,
    seqused_k: Any = None,
    page_table: Any = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    qv: Any = None,
    q_descale: Any = None,
    k_descale: Any = None,
    v_descale: Any = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Any = None,
    only_qv: bool = False,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: Any = None,
    score_mod: Any = None,
    aux_tensors: Any = None,
    sfq: Any = None,
    sfk: Any = None,
    sfv: Any = None,
    rel_bias: Any = None,
    rel_bias_prep_cache: Any = None,
    ver: int = 3,
    out: torch.Tensor | None = None,
    **_unused: Any,
) -> torch.Tensor:
    """Drop-in replacement for SGLang's varlen entry point, on SDPA.

    The signature mirrors upstream's exactly, including the arguments we
    refuse, because callers pass them by keyword and a missing parameter would
    surface as an unrelated ``TypeError`` far from the real cause.

    ``ver`` is accepted and ignored: it selects between FlashAttention 3 and
    4, and neither exists here.

    Shapes follow the packed varlen convention:
    ``q`` is ``[total_q, heads, dim]``, ``k``/``v`` are
    ``[total_k, kv_heads, dim]``, and the result matches ``q``.
    """
    _reject_unsupported(
        window_size=window_size, softcap=softcap,
        return_softmax_lse=return_softmax_lse,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        sinks=sinks, score_mod=score_mod, qv=qv, page_table=page_table,
        seqused_q=seqused_q, seqused_k=seqused_k,
    )
    if attention_chunk:
        raise VarlenFeatureUnsupported(f"attention_chunk={attention_chunk!r}")

    num_heads = q.shape[1]
    k = _expand_kv_heads(k, num_heads)
    v = _expand_kv_heads(v, num_heads)

    lens_q = _segment_lengths(cu_seqlens_q)
    lens_k = _segment_lengths(cu_seqlens_k)
    if len(lens_q) != len(lens_k):
        raise ValueError(
            f"cu_seqlens_q describes {len(lens_q)} segments but cu_seqlens_k "
            f"describes {len(lens_k)}"
        )

    result = torch.empty_like(q) if out is None else out

    # Uniform-length fast path. The hybrid-window backend builds its segments
    # from "consecutive window frames with identical bounds", so in practice
    # almost every call lands here -- and here the whole varlen call is one
    # SDPA invocation with a reshape, no Python loop and no per-segment
    # launch overhead.
    if lens_q and len(set(lens_q)) == 1 and len(set(lens_k)) == 1:
        n, lq, lk = len(lens_q), lens_q[0], lens_k[0]
        if lq and lk:
            qb = q.reshape(n, lq, num_heads, -1).transpose(1, 2)
            kb = k.reshape(n, lk, num_heads, -1).transpose(1, 2)
            vb = v.reshape(n, lk, num_heads, -1).transpose(1, 2)
            attn = _sdpa(qb, kb, vb, scale=softmax_scale, causal=causal)
            result.copy_(attn.transpose(1, 2).reshape(n * lq, num_heads, -1))
            return result

    # Ragged path: one dense call per segment.
    qo = 0
    ko = 0
    for lq, lk in zip(lens_q, lens_k):
        if lq == 0:
            ko += lk
            continue
        if lk == 0:
            # No keys to attend to. FlashAttention writes zeros here; leaving
            # `result` untouched would hand back uninitialised memory from
            # `torch.empty_like`, which reads as plausible noise.
            result[qo:qo + lq].zero_()
            qo += lq
            continue
        attn = _sdpa(
            q[qo:qo + lq].transpose(0, 1).unsqueeze(0),
            k[ko:ko + lk].transpose(0, 1).unsqueeze(0),
            v[ko:ko + lk].transpose(0, 1).unsqueeze(0),
            scale=softmax_scale, causal=causal,
        )
        result[qo:qo + lq].copy_(attn.squeeze(0).transpose(0, 1))
        qo += lq
        ko += lk

    return result


__all__ = [
    "VarlenFeatureUnsupported",
    "flash_attn_varlen_func",
    "SDPA_BACKEND_OBSERVED",
]
