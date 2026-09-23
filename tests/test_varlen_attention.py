"""Tests for the RDNA3 varlen attention primitive.

These are the tests that matter most in this repository. Every other shim
fails loudly when it is wrong -- an ImportError, a compile error, a refusal to
start. This one computes numbers, and a wrong attention output is *silent*:
the model still produces a video, it is merely worse, and nothing between the
kernel and the pixels reports anything. So the correctness tests here compare
against an explicit reference softmax rather than against another SDPA call,
and they cover the ragged path as well as the uniform fast path -- the two
take completely different branches and only one of them is exercised by
typical VDN-H3 traffic.

No GPU required: SDPA runs on CPU, and the decomposition being tested is
device-independent.
"""

import pytest

torch = pytest.importorskip("torch", reason="varlen shim needs torch")

from sglang_radeon_rdna3 import varlen_attention as va  # noqa: E402


def _reference(q, k, v, lens_q, lens_k, scale=None, causal=False):
    """Packed varlen attention, written out longhand.

    Deliberately *not* implemented with SDPA. If both sides called SDPA the
    test would only prove the reshaping is self-consistent, which is not the
    property at risk.
    """
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale
    out = torch.empty_like(q)
    qo = ko = 0
    for lq, lk in zip(lens_q, lens_k):
        if lq == 0:
            ko += lk
            continue
        if lk == 0:
            out[qo:qo + lq].zero_()
            qo += lq
            continue
        # [heads, lq, dim] x [heads, dim, lk] -> [heads, lq, lk]
        qs = q[qo:qo + lq].transpose(0, 1).float()
        ks = k[ko:ko + lk].transpose(0, 1).float()
        vs = v[ko:ko + lk].transpose(0, 1).float()
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        if causal:
            mask = torch.ones(lq, lk, dtype=torch.bool).tril(diagonal=lk - lq)
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[qo:qo + lq] = torch.matmul(probs, vs).transpose(0, 1).to(out.dtype)
        qo += lq
        ko += lk
    return out


def _cu(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)),
                        dtype=torch.int32)


def _pack(lens_q, lens_k, heads=4, dim=16, kv_heads=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv_heads = kv_heads or heads
    q = torch.randn(sum(lens_q), heads, dim, generator=g)
    k = torch.randn(sum(lens_k), kv_heads, dim, generator=g)
    v = torch.randn(sum(lens_k), kv_heads, dim, generator=g)
    return q, k, v


def _call(q, k, v, lens_q, lens_k, **kw):
    return va.flash_attn_varlen_func(
        q, k, v, _cu(lens_q), _cu(lens_k),
        max_seqlen_q=max(lens_q), max_seqlen_k=max(lens_k), **kw,
    )


# --- correctness: the two branches, against a longhand reference -----------

def test_uniform_segments_match_the_reference():
    """The fast path: equal-length segments, one batched SDPA, no loop.

    This is the branch almost every VDN-H3 call takes, because the hybrid
    window backend groups "consecutive window frames with identical bounds"
    into one varlen segment.
    """
    lens_q = lens_k = [8, 8, 8]
    q, k, v = _pack(lens_q, lens_k)
    got = _call(q, k, v, lens_q, lens_k)
    want = _reference(q, k, v, lens_q, lens_k)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_ragged_segments_match_the_reference():
    """The loop path -- and the one real traffic exercises least.

    A bug that only shows up when segment lengths differ would survive every
    smoke test and then corrupt exactly the frames at window boundaries.
    """
    lens_q = [5, 1, 9, 3]
    lens_k = [5, 1, 9, 3]
    q, k, v = _pack(lens_q, lens_k, seed=1)
    got = _call(q, k, v, lens_q, lens_k)
    want = _reference(q, k, v, lens_q, lens_k)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_query_and_key_segments_may_differ_in_length():
    """Cross-attention shape: cu_seqlens_q is not cu_seqlens_k.

    The decomposed window planner builds exactly this -- a short run of query
    rows against a large gathered key set -- so getting the two offsets
    independent is load-bearing, and swapping them would still run.
    """
    lens_q = [2, 4]
    lens_k = [7, 3]
    q, k, v = _pack(lens_q, lens_k, seed=2)
    got = _call(q, k, v, lens_q, lens_k)
    want = _reference(q, k, v, lens_q, lens_k)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_single_segment_is_plain_dense_attention():
    lens_q = lens_k = [12]
    q, k, v = _pack(lens_q, lens_k, seed=3)
    got = _call(q, k, v, lens_q, lens_k)
    want = _reference(q, k, v, lens_q, lens_k)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_explicit_softmax_scale_is_honoured():
    """A scale silently replaced by the 1/sqrt(d) default is invisible.

    The output stays finite and well-shaped; only the sharpness of the
    attention changes. Pick a scale far from the default so the reference
    disagrees loudly if it is dropped.
    """
    lens_q = lens_k = [6, 6]
    q, k, v = _pack(lens_q, lens_k, seed=4)
    scale = 0.037
    got = _call(q, k, v, lens_q, lens_k, softmax_scale=scale)
    want = _reference(q, k, v, lens_q, lens_k, scale=scale)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    # and it really is different from the default, or the test proves nothing
    default = _call(q, k, v, lens_q, lens_k)
    assert not torch.allclose(got, default, rtol=1e-2, atol=1e-2)


def test_causal_is_supported_for_the_dense_fallback():
    """VDN-H3 never asks for causal, but FlashAttentionImpl.forward_varlen
    serves other models through the same symbol. Support it rather than let
    an unrelated model get non-causal attention silently."""
    lens_q = lens_k = [7, 7]
    q, k, v = _pack(lens_q, lens_k, seed=5)
    got = _call(q, k, v, lens_q, lens_k, causal=True)
    want = _reference(q, k, v, lens_q, lens_k, causal=True)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(got, _call(q, k, v, lens_q, lens_k),
                              rtol=1e-2, atol=1e-2)


# --- shapes and edges ------------------------------------------------------

def test_grouped_query_heads_are_broadcast():
    lens_q = lens_k = [4, 4]
    q, k, v = _pack(lens_q, lens_k, heads=8, kv_heads=2, seed=6)
    got = _call(q, k, v, lens_q, lens_k)
    k_exp = k.repeat_interleave(4, dim=1)
    v_exp = v.repeat_interleave(4, dim=1)
    want = _reference(q, k_exp, v_exp, lens_q, lens_k)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_head_counts_that_do_not_divide_are_refused():
    lens_q = lens_k = [4]
    q, k, v = _pack(lens_q, lens_k, heads=6, kv_heads=4, seed=7)
    with pytest.raises(va.VarlenFeatureUnsupported):
        _call(q, k, v, lens_q, lens_k)


def test_empty_key_segment_yields_zeros_not_garbage():
    """`torch.empty_like` is uninitialised memory; skipping a segment without
    writing it hands back whatever was on the heap, which reads as plausible
    noise rather than as an error."""
    lens_q = [3, 3]
    lens_k = [4, 0]
    q, k, v = _pack(lens_q, lens_k, seed=8)
    got = _call(q, k, v, lens_q, lens_k)
    assert torch.all(got[3:] == 0)
    torch.testing.assert_close(got[:3], _reference(q, k, v, [3], [4])[:3],
                               rtol=1e-4, atol=1e-4)


def test_out_is_written_in_place_and_returned():
    """The window planner passes slices of a preallocated tensor as `out=` and
    relies on the write landing there, discarding the return value."""
    lens_q = lens_k = [5, 5]
    q, k, v = _pack(lens_q, lens_k, seed=9)
    dest = torch.full_like(q, float("nan"))
    got = _call(q, k, v, lens_q, lens_k, out=dest)
    assert got.data_ptr() == dest.data_ptr()
    assert not torch.isnan(dest).any()
    torch.testing.assert_close(dest, _reference(q, k, v, lens_q, lens_k),
                               rtol=1e-4, atol=1e-4)


def test_mismatched_segment_counts_are_an_error():
    q, k, v = _pack([4, 4], [4, 4], seed=10)
    with pytest.raises(ValueError, match="segments"):
        va.flash_attn_varlen_func(q, k, v, _cu([4, 4]), _cu([8]))


# --- refusals --------------------------------------------------------------
#
# Each of these is a feature real FlashAttention has and this shim does not.
# Ignoring any one of them produces plausible-looking but wrong attention,
# which is the failure mode this module exists to avoid.

@pytest.mark.parametrize("kwargs", [
    {"window_size": (128, 0)},
    {"softcap": 30.0},
    {"return_softmax_lse": True},
    {"q_descale": 1.0},
    {"k_descale": 1.0},
    {"v_descale": 1.0},
    {"sinks": 1},
    {"score_mod": lambda *a: a},
    {"qv": 1},
    {"page_table": 1},
    {"seqused_q": 1},
    {"seqused_k": 1},
    {"attention_chunk": 64},
])
def test_unsupported_features_raise_rather_than_being_ignored(kwargs):
    lens_q = lens_k = [4]
    q, k, v = _pack(lens_q, lens_k, seed=11)
    with pytest.raises(va.VarlenFeatureUnsupported):
        _call(q, k, v, lens_q, lens_k, **kwargs)


def test_ver_is_accepted_and_ignored():
    """`ver` selects FlashAttention 3 vs 4 upstream; callers pass it
    unconditionally and neither exists here."""
    lens_q = lens_k = [4]
    q, k, v = _pack(lens_q, lens_k, seed=12)
    a = _call(q, k, v, lens_q, lens_k, ver=3)
    b = _call(q, k, v, lens_q, lens_k, ver=4)
    torch.testing.assert_close(a, b)


def test_window_size_minus_one_is_not_a_window():
    """(-1, -1) is FlashAttention's "no window" sentinel, and every caller
    passes it. Refusing it would reject the entire supported path."""
    lens_q = lens_k = [4]
    q, k, v = _pack(lens_q, lens_k, seed=13)
    _call(q, k, v, lens_q, lens_k, window_size=(-1, -1))
