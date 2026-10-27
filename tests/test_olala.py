"""Olala model: fused Metal decode vs the reference path, cache semantics, weight merging.

Uses a tiny randomly initialised configuration, no checkpoint needed.
"""

import copy
import importlib
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_engine.models.olala import metal, register
from mlx_engine.models.olala import model as olala_model

register()

TINY = {
    "model_type": "olala",
    "vocab_size": 512,
    "hidden_size": 256,
    "num_hidden_layers": 5,
    "layers_config": "MVMMV",
    "num_attention_heads": 8,
    "num_signal_heads_diff": 6,
    "head_dim": 128,
    "moe_num_routed_experts": 32,
    "moe_num_active_experts": 6,
    "moe_routed_input_dim": 64,
    "moe_routed_intermediate_size": 128,
    "moe_shared_intermediate_size": 256,
    "max_position_embeddings": 4096,
    "slw_wsize": 4096,
}


def _tiny_model(seed=0, dtype=mx.float32):
    mx.random.seed(seed)
    args = olala_model.ModelArgs.from_dict(TINY)
    m = olala_model.Model(args)
    # small random weights; norms / biases around their trained ranges
    params = tree_map(lambda p: mx.random.normal(p.shape) * 0.05, m.parameters())
    m.update(params)
    for layer in m.model.layers:
        layer.geodesic_mixer.scale = mx.array([1.0])
        layer.geodesic_mixer.bias = mx.array([0.05])
        layer.geodesic_mlp.scale = mx.array([1.0])
        layer.geodesic_mlp.bias = mx.array([0.05])
    m.update(tree_map(lambda p: p.astype(dtype), m.parameters()))
    mx.eval(m.parameters())
    return m


def _decode_logits(m, prompt, steps, fused, monkeypatch):
    monkeypatch.setenv("OLALA_FUSED", "1" if fused else "0")
    cache = m.make_cache()
    logits = m(prompt, cache=cache)
    tok = mx.argmax(logits[:, -1], axis=-1).reshape(-1, 1)
    out = []
    for _ in range(steps):
        logits = m(tok, cache=cache).astype(mx.float32)[:, -1]
        out.append(logits)
        tok = mx.argmax(logits, axis=-1).reshape(-1, 1)
    return mx.stack(out, axis=1)


@pytest.mark.parametrize("batch", [1, 3])
def test_fused_decode_matches_reference(monkeypatch, batch):
    monkeypatch.setattr(olala_model, "_ACT_F32", True)
    m = _tiny_model()
    prompt = mx.random.randint(0, TINY["vocab_size"], (batch, 21))
    ref = _decode_logits(m, prompt, 5, fused=False, monkeypatch=monkeypatch)
    monkeypatch.setenv("OLALA_FUSED", "1")
    assert metal.can_decode(
        m.model, mx.zeros((batch, 1, 256)), _primed_cache(m, prompt), None
    )
    fused = _decode_logits(m, prompt, 5, fused=True, monkeypatch=monkeypatch)
    assert mx.allclose(ref, fused, atol=2e-3, rtol=2e-3).item(), (
        mx.abs(ref - fused).max().item()
    )


def _primed_cache(m, prompt):
    cache = m.make_cache()
    mx.eval(m(prompt, cache=cache))
    return cache


def test_chunked_prefill_matches_stepwise(monkeypatch):
    monkeypatch.setenv("OLALA_FUSED", "0")
    m = _tiny_model()
    prompt = mx.random.randint(0, TINY["vocab_size"], (1, 45))
    full = m(prompt).astype(mx.float32)
    cache = m.make_cache()
    parts = [
        m(prompt[:, s : s + 13], cache=cache).astype(mx.float32)
        for s in range(0, 45, 13)
    ]
    assert mx.allclose(full, mx.concatenate(parts, axis=1), atol=1e-4).item()
    cache = m.make_cache()
    steps = [m(prompt[:, i : i + 1], cache=cache).astype(mx.float32) for i in range(45)]
    assert mx.allclose(full, mx.concatenate(steps, axis=1), atol=1e-4).item()


def test_m3_chunked_matches_sequential():
    m = _tiny_model().layers[0].mixer
    B, L = 3, 37
    proj = mx.random.normal((B, L, m.n_static + m.n_dyn))
    sw = m.P * m.N + m.R * m.N + m.P + m.A
    st = mx.random.normal((B, m.H, sw)) * 0.1
    rng = mx.array([[0, L], [3, L - 6], [5, 5]], dtype=mx.int32)
    prm = metal.m3_param_pack(m)
    args = (prm, st, m.H, m.P, m.N, m.R, m.A, m.n_static)
    y0, s0 = metal.m3_prefill(proj, *args, rng=rng)
    y1, s1 = metal.m3_chunked(proj, *args, rng=rng)
    assert mx.allclose(s0, s1, atol=1e-4, rtol=1e-4).item()
    for b, (a, e) in enumerate(rng.tolist()):
        assert mx.allclose(y0[b, a:e], y1[b, a:e], atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("T", [1, 3])
def test_gemv_split_rows_match_reference(T):
    # an f32 segment next to a quantized one: the f32 rows are split across simdgroups
    K = 1536
    w32 = mx.random.normal((256, K)) * 0.05
    wq = mx.random.normal((640, K)) * 0.05
    q, s, b = mx.quantize(wq, group_size=64, bits=6)
    ws = [metal.W(w32), metal.W(q, s, b, 6, 64)]
    assert metal._gemv_splits([(w.fmt, w.gs or 0, w.rows) for w in ws], K, 8)[0] > 1
    x = mx.random.normal((T, K))
    xr = mx.random.normal((T, 640))
    y32, yq = metal.gemv([x], ws, T=T)
    assert mx.allclose(y32, x @ w32.T, atol=1e-4).item()
    wdq = mx.dequantize(q, s, b, group_size=64, bits=6)
    assert mx.allclose(yq, x @ wdq.T, atol=1e-4).item()
    # geodesic sums of a split GEMV
    y, gst = metal.gemv([x], [ws[1]], T=T, stats_x=xr)
    ref = mx.stack([(xr * xr).sum(-1), (xr * y).sum(-1), (y * y).sum(-1)], -1)
    assert mx.allclose(gst[: 4 * T].reshape(T, 4)[:, :3], ref, rtol=1e-4).item()


@pytest.mark.parametrize(
    "T,L,window", [(37, 37, 4096), (300, 133, 4096), (300, 300, 100)]
)
def test_attn_prefill_mma_matches_scalar(monkeypatch, T, L, window):
    q = (mx.random.normal((1, 8, L, 128)) * 3).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, T, 128)) * 3).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, T, 128)).astype(mx.bfloat16)
    a = metal.attn_prefill_mma(q, k, v, 1 / 128, 150.0, window)
    monkeypatch.setattr(metal, "_FA_MMA", False)
    b = metal.attn_prefill(q, k, v, 1 / 128, 150.0, window)
    assert mx.allclose(a.astype(mx.float32), b.astype(mx.float32), atol=1e-2).item()


@pytest.mark.parametrize("S,kstart", [(5, 0), (70, 3), (300, 0)])
def test_attn_decode_short_matches_long(S, kstart):
    T, Hq, Hk, Dh = 2, 16, 4, 128
    keys = (mx.random.normal((T, Hk, S + 9, Dh)) * 2).astype(mx.bfloat16)
    values = mx.random.normal((T, Hk, S + 9, Dh)).astype(mx.bfloat16)
    q = mx.random.normal((T, Hq, Dh)) * 3
    ks = mx.array([kstart, 0], dtype=mx.int32)
    args = (q, keys, values, ks, 1 / 128, 150.0, T, Hq, Hk, Dh)
    slen = mx.array([S], dtype=mx.int32)

    def merged(o, m, l):
        M = m.max(axis=-1, keepdims=True)
        c = mx.where(m == -mx.inf, 0, mx.exp(m - M))
        return (o * c[..., None]).sum(-2) / (l * c).sum(-1)[..., None]

    short = merged(*metal.attn_decode_short(*args, S, slen))
    old = metal.ATT_SHORT_MAX
    metal.ATT_SHORT_MAX = 0
    try:
        long = merged(*metal.attn_decode(*args, S=S, slen=slen))
    finally:
        metal.ATT_SHORT_MAX = old
    assert mx.allclose(short, long, atol=1e-4).item()


def test_cache_layout_one_entry_per_layer():
    m = _tiny_model()
    cache = m.make_cache()
    assert len(cache) == len(m.layers)
    for layer, c in zip(m.layers, cache, strict=True):
        if layer.layer_type == "M":
            assert type(c).__name__ == "ArraysCache"
        else:
            assert type(c).__name__ == "CacheList"
            assert type(c[0]).__name__ == "KVCache"
    mx.eval(m(mx.zeros((1, 7), mx.int32), cache=cache))
    assert cache[1].offset == 7


def test_kv_inplace_append_is_disabled_on_shared_buffers():
    c = olala_model.KVCache()
    c.update_and_fetch(mx.zeros((1, 2, 5, 128)), mx.zeros((1, 2, 5, 128)))
    assert c.can_append_inplace(1)
    snap = copy.deepcopy(c)
    # both share the buffer now: neither may write in place until it has its own
    assert not c.can_append_inplace(1) and not snap.can_append_inplace(1)
    c.update_and_fetch(mx.ones((1, 2, 1, 128)), mx.ones((1, 2, 1, 128)))
    mx.eval(c.keys, snap.keys)
    assert c.can_append_inplace(1)
    assert snap.offset == 5 and mx.all(snap.keys[..., :5, :] == 0).item()
    # trimming below the high-water mark disables in-place appends
    c.trim(2)
    assert not c.can_append_inplace(1)


def test_sanitize_merges_and_is_idempotent():
    m = _tiny_model()
    flat = dict(tree_flatten(m.parameters()))
    # split the merged matrices back into the HF layout
    hf = {}
    d = 256
    for k, v in flat.items():
        if (
            k.endswith("mixer.in_proj.weight")
            and v.shape[0] == m.layers[0].mixer.n_static + m.layers[0].mixer.n_dyn
        ):
            n = m.layers[0].mixer.n_static
            hf[k] = v[:n]
            hf[k.replace("in_proj", "in_proj_dyn")] = v[n:]
        elif k.endswith("mlp.in_cat.weight"):
            hf[k.replace("in_cat", "down_proj")] = v[:64]
            hf[k.replace("in_cat", "shared_experts.fc_1")] = v[64:]
        elif k.endswith("mlp.out_cat.weight"):
            hf[k.replace("out_cat", "up_proj")] = v[:, :64]
            hf[k.replace("out_cat", "shared_experts.fc_2")] = v[:, 64:]
        else:
            hf[k] = v
    merged = m.sanitize(hf)
    again = m.sanitize(dict(merged))
    assert set(merged) == set(again)
    for k, v in flat.items():
        if k.endswith("mixer.in_proj.weight") and v.shape[0] != d and k in merged:
            assert mx.array_equal(merged[k], v).item()
    assert mx.array_equal(
        merged["model.layers.0.mlp.in_cat.weight"],
        flat["model.layers.0.mlp.in_cat.weight"],
    ).item()
    assert mx.array_equal(
        merged["model.layers.0.mlp.out_cat.weight"],
        flat["model.layers.0.mlp.out_cat.weight"],
    ).item()


def test_registered_for_mlx_lm_and_mlx_vlm():
    assert importlib.import_module("mlx_lm.models.olala").Model is olala_model.Model
    vlm = importlib.import_module("mlx_vlm.models.olala")
    cfg = vlm.ModelConfig.from_dict(dict(TINY, unknown_key=1))
    model = vlm.Model(cfg)
    assert len(model.make_cache()) == TINY["num_hidden_layers"]


@pytest.mark.skipif(os.environ.get("OLALA_TEST_QUANT") == "0", reason="disabled")
@pytest.mark.parametrize("bits", [8, 6, 5, 4])
def test_fused_decode_quantized(monkeypatch, bits):
    monkeypatch.setattr(olala_model, "_ACT_F32", True)
    from mlx import nn

    m = _tiny_model()
    nn.quantize(
        m,
        group_size=64,
        bits=bits,
        class_predicate=lambda p, mod: (
            m.quant_predicate(p, mod) and hasattr(mod, "to_quantized")
        ),
    )
    prompt = mx.random.randint(0, TINY["vocab_size"], (2, 17))
    ref = _decode_logits(m, prompt, 4, fused=False, monkeypatch=monkeypatch)
    fused = _decode_logits(m, prompt, 4, fused=True, monkeypatch=monkeypatch)
    assert mx.allclose(ref, fused, atol=5e-3, rtol=5e-3).item(), (
        mx.abs(ref - fused).max().item()
    )
