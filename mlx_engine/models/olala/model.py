"""Olala 7A1B (OlalaForCausalLM) for mlx-lm / mlx-vlm.

Hybrid model: 30 Mamba-3 MIMO layers ("M") and 6 differential tensor-product
attention layers ("V"), a 256-expert MoE (6 active, latent 384) on every
layer, and a geodesic (rotation) residual update instead of pre-norms.

Reference: modeling_olala.py shipped with the checkpoints, and the llama.cpp
implementation (src/models/olala.cpp) it was validated against.

Projections that read the same input are merged into one matrix at load time
(``Model.sanitize``), so that decoding streams each layer's weights in a few
large matrix-vector products:
  - M mixer ``in_proj``: [in_proj (z|x|dt|A|trap per head) | in_proj_dyn (B|C|angles)]
  - V mixer ``in_proj``: [c_q | W_B_k | W_B_v | gate_proj], and ``in_proj_small``:
    [W_A_k | W_A_v | shift_proj_k | shift_proj_v | lambda_proj] (kept unquantized)
  - MoE ``in_cat``: [down_proj | shared fc_1], ``out_cat``: [up_proj | shared fc_2]
    (concatenated along the input dimension)

Cache layout (one entry per layer):
  - type M: ArraysCache(1) holding the packed f32 state (B, H, P*N + R*N + P + A),
    per head [ssm S (P,N) | rotated keys of the last token (R,N) | raw x of the last token (P) |
    rotary angles (A)]
  - type V: CacheList(KVCache (keys after token shift and k-norm), ArraysCache(2) = raw k / v
    of the last token, the token-shift input)
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.cache import CacheList as _MlxCacheList
from mlx_lm.models.cache import KVCache as _MlxKVCache
from mlx_lm.models.switch_layers import SwitchLinear

from . import kernels

# compute every matmul input in f32 (slow with bf16 weights; for accuracy checks)
_ACT_F32 = os.environ.get("OLALA_ACT_F32", "0") == "1"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "olala"
    vocab_size: int = 151936
    hidden_size: int = 1536
    num_hidden_layers: int = 36
    layers_config: str = "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM"
    norm_epsilon: float = 1e-6
    zero_centered_gamma: bool = True
    # attention (V layers)
    num_attention_heads: int = 48
    num_signal_heads_diff: int = 36
    head_dim: int = 128
    tpa_rank: int = 4
    softcap_attn: float = 150.0
    slw_wsize: int = 32768
    scalable_softmax: bool = True
    token_shift_attn: bool = True
    qk_norm: bool = True
    gate_attn: bool = True
    zero_centered_gate: bool = True
    use_completed_p: bool = True
    # Mamba-3 (M layers)
    mamba_d_state: int = 128
    mamba_headdim: int = 64
    mamba_mimo_dim: int = 4
    mamba_ngroups: int = 1
    # MoE
    moe_num_routed_experts: int = 256
    moe_num_active_experts: int = 6
    moe_routed_input_dim: int = 384
    moe_routed_intermediate_size: int = 768
    moe_routed_scaling_factor: float = 2.45
    moe_shared_intermediate_size: int = 1536
    # misc
    normalize_embeddings_ngpt: bool = True
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 65536

    def __post_init__(self):
        unsupported = set(self.layers_config) - set("MV")
        if unsupported:
            raise ValueError(f"Olala: unsupported layer types {sorted(unsupported)}")
        if len(self.layers_config) != self.num_hidden_layers:
            raise ValueError("Olala: layers_config does not match num_hidden_layers")
        if not (
            self.token_shift_attn
            and self.qk_norm
            and self.gate_attn
            and self.scalable_softmax
        ):
            raise ValueError(
                "Olala: only the released attention configuration is supported"
            )


class InplaceAppendMixin:
    """KVCache mixin that lets the fused decode path append to the cache in place.

    mlx-lm / mlx-vlm append with a slice update; while the previous token's command
    buffer is in flight MLX keeps a reference to the buffer, so every append copies
    the whole cache (measured 0.85 ms / token at 8k context for the 7 attention layers).

    In-place writes are only allowed at positions no one else can see: positions at
    or above the high-water mark ``_hw`` of everything ever exposed (returned views,
    ``state``). Snapshots read only below their own offset (<= _hw). Copies
    (``copy`` / ``deepcopy`` share the MLX buffer) and restored states disable
    in-place appends until the next regular append has given the cache its own
    buffer (copy-on-write)."""

    _hw = 0

    def update_and_fetch(self, keys, values):
        out = super().update_and_fetch(keys, values)
        self._hw = self.offset
        return out

    @property
    def state(self):
        return super().state

    @state.setter
    def state(self, v):
        super(InplaceAppendMixin, type(self)).state.fset(self, v)
        self._hw = self.keys.shape[2] if self.keys is not None else 0

    def __copy__(self):
        return self._shared_copy()

    def __deepcopy__(self, memo):
        return self._shared_copy()

    def _shared_copy(self):
        c = self.__class__.__new__(self.__class__)
        c.__dict__.update(self.__dict__)
        if self.keys is not None:
            self._hw = c._hw = self.keys.shape[2]
        return c

    def can_append_inplace(self, n: int = 1) -> bool:
        return (
            self.keys is not None
            and isinstance(self.offset, int)
            and self.offset >= self._hw
            and self.offset + n <= self.keys.shape[2]
        )

    def commit_inplace(self, n: int = 1):
        self.offset += n
        self._hw = self.offset


class KVCache(InplaceAppendMixin, _MlxKVCache):
    """mlx-lm KVCache with in-place appends for the fused decode path."""


class AttnCacheMixin:
    """Cache of one attention layer: [KV cache, ArraysCache(2) = raw k / v of the last
    token (token-shift input)]. One entry per layer keeps ``len(cache) == len(layers)``,
    which mlx-engine relies on; ``offset`` / ``keys`` expose the KV cache."""

    @property
    def offset(self):
        return self.caches[0].offset

    @property
    def keys(self):
        return self.caches[0].keys


class CacheList(AttnCacheMixin, _MlxCacheList):
    pass


def _zc(weight: mx.array, zero_centered: bool) -> mx.array:
    return 1.0 + weight if zero_centered else weight


class _Norm(nn.Module):
    """RMSNorm weight stored as ``<name>.norm.weight`` (checkpoint layout)."""

    def __init__(self, dims: int):
        super().__init__()
        self.norm = nn.Module()
        self.norm.weight = mx.zeros((dims,))


class _Geodesic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = mx.ones((1,))
        self.bias = mx.zeros((1,))


def geodesic_update(x: mx.array, g: mx.array, scale, bias, layer_idx: int) -> mx.array:
    """Rotate the residual x towards the update g (OlalaGeodesicNorm), in f32."""
    g = g.astype(mx.float32)
    x_norm_sq = mx.maximum(mx.sum(x * x, axis=-1, keepdims=True), 1e-12)
    proj = mx.sum(x * g, axis=-1, keepdims=True) / x_norm_sq
    grad = g - proj * x
    tn = mx.maximum(mx.sqrt(mx.sum(grad * grad, axis=-1, keepdims=True)), 1e-8)
    r = mx.maximum(mx.sqrt(x_norm_sq), 1e-6)
    clamp = math.pi / 4
    theta = mx.minimum(tn / r, clamp)
    theta = mx.minimum((theta * scale + bias) / (layer_idx + 1), clamp)
    return x * mx.cos(theta) + grad * (r / tn) * mx.sin(theta)


class Mamba3Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.hidden_size
        self.d_inner = 2 * args.hidden_size
        self.P = args.mamba_headdim
        self.H = self.d_inner // self.P
        self.N = args.mamba_d_state
        self.R = args.mamba_mimo_dim
        split = int(self.N * 0.5)
        split -= split % 2
        self.A = split // 2  # number of rotary angles
        self.eps = args.norm_epsilon
        self.zc = args.zero_centered_gamma
        if args.mamba_ngroups != 1:
            raise ValueError("Olala: mamba_ngroups != 1 is not supported")

        H, R, N, P = self.H, self.R, self.N, self.P
        self.n_static = self.d_inner * 2 + 3 * H  # z | x | dt | A | trap per head
        self.n_dyn = 2 * R * N + self.A  # B | C | angles
        self.in_proj = nn.Linear(self.d_model, self.n_static + self.n_dyn, bias=False)
        self.B_bias = mx.ones((H, R, N))
        self.C_bias = mx.ones((H, R, N))
        self.B_norm = _Norm(N)
        self.C_norm = _Norm(N)
        self.in_proj_mimo_x = mx.ones((H, R, P))
        self.in_proj_mimo_z = mx.ones((H, R, P))
        self.out_proj_mimo = mx.ones((H, R, P))
        self.dt_bias = mx.zeros((H,))
        self.D = mx.ones((H,))

    def __call__(self, h: mx.array, cache: ArraysCache | None, mask: mx.array | None):
        L = h.shape[1]
        out = self.in_proj(h)
        from . import metal

        if (
            L > 1
            and metal.enabled()
            and os.environ.get("OLALA_FUSED_PREFILL", "1") == "1"
        ):
            return self._fused_prefill(out, cache)
        proj, dyn = out[..., : self.n_static], out[..., self.n_static :]
        state = None
        if cache is not None and cache[0] is not None:
            state = kernels.m3_unpack(cache[0], self)
        y, new_state = kernels.mamba3_forward(
            proj,
            dyn,
            state,
            self,
            mask,
            lengths=None if cache is None else getattr(cache, "lengths", None),
        )
        if cache is not None:
            cache[0] = kernels.m3_pack(new_state, self)
            if hasattr(cache, "advance"):
                cache.advance(L)
        return y

    def _fused_prefill(self, out, cache):
        """Sequential Metal kernel over the prompt (metal.m3_prefill)."""
        from . import metal

        Bsz, L, _ = out.shape
        if not hasattr(self, "_prm"):
            self._prm = metal.m3_param_pack(self)
            mx.eval(self._prm)
        sw = self.P * self.N + self.R * self.N + self.P + self.A
        st = (
            cache[0]
            if cache is not None and cache[0] is not None
            else mx.zeros((Bsz, self.H, sw), mx.float32)
        )
        rng = None
        if cache is not None:
            lp = getattr(cache, "left_padding", None)
            lengths = getattr(cache, "lengths", None)
            if lp is not None or lengths is not None:
                s0 = (
                    mx.zeros((Bsz,), mx.int32)
                    if lp is None
                    else mx.maximum(lp, 0).astype(mx.int32)
                )
                s1 = (
                    mx.full((Bsz,), L, mx.int32)
                    if lengths is None
                    else mx.minimum(lengths, L).astype(mx.int32)
                )
                rng = mx.stack([s0, mx.maximum(s1, s0)], axis=1)
        y, st = metal.m3_prefill(
            out.astype(mx.float32),
            self._prm,
            st.astype(mx.float32),
            self.H,
            self.P,
            self.N,
            self.R,
            self.A,
            self.n_static,
            rng=rng,
        )
        if cache is not None:
            cache[0] = st
            if hasattr(cache, "advance"):
                cache.advance(L)
        return y.astype(out.dtype) if out.dtype != mx.float32 else y


class DiffTPAttention(nn.Module):
    """Differential attention V2 with tensor-product K/V (rank 4), token shift,
    scalable softmax, soft-capped logits and an output gate."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        d = args.hidden_size
        self.layer_idx = layer_idx
        self.n_heads = args.num_attention_heads
        self.n_sig = args.num_signal_heads_diff or self.n_heads // 2
        self.n_noise = self.n_heads - self.n_sig
        self.n_kv = self.n_noise
        self.snr = self.n_sig // self.n_noise
        self.head_dim = args.head_dim
        self.rank = args.tpa_rank
        self.softcap = args.softcap_attn
        self.window = args.slw_wsize
        self.eps = args.norm_epsilon
        self.zc = args.zero_centered_gamma
        self.gate_bias = 1.15 if args.zero_centered_gate else 0.0
        self.scale = (
            1.0 / self.head_dim if args.use_completed_p else self.head_dim**-0.5
        )

        Hq, Hk, Dh, Rk = self.n_heads, self.n_kv, self.head_dim, self.rank
        # [c_q | W_B_k | W_B_v | gate_proj]
        self.split_big = [Hq * Dh, Rk * Dh, Rk * Dh, self.n_sig * Dh]
        # [W_A_k | W_A_v | shift_proj_k | shift_proj_v | lambda_proj]
        self.split_small = [Hk * Rk, Hk * Rk, Hk, Hk, self.n_noise]
        self.in_proj = nn.Linear(d, sum(self.split_big), bias=False)
        self.in_proj_small = nn.Linear(d, sum(self.split_small), bias=False)
        self.q_norm = _Norm(Dh)
        self.k_norm = _Norm(Dh)
        self.softmax_scaler = mx.ones((Hq,))

    def project(self, h):
        """Returns q (B,L,Hq,Dh), k_raw / v_raw (B,L,Hk,Dh), alpha_k / alpha_v (B,L,Hk,1),
        lambda (B,L,Hn,1,1), gate (B,L,Hs,Dh); all f32."""
        B, L, _ = h.shape
        Hq, Hk, Dh, Rk = self.n_heads, self.n_kv, self.head_dim, self.rank
        big = self.in_proj(h).astype(mx.float32)
        small = self.in_proj_small(h.astype(self.in_proj_small.weight.dtype)).astype(
            mx.float32
        )
        q, b_k, b_v, gate = mx.split(big, list(_cumsum(self.split_big[:-1])), axis=-1)
        a_k, a_v, s_k, s_v, lam = mx.split(
            small, list(_cumsum(self.split_small[:-1])), axis=-1
        )
        q = q.reshape(B, L, Hq, Dh)
        k = (a_k.reshape(B, L, Hk, Rk) @ b_k.reshape(B, L, Rk, Dh)) / Rk
        v = (a_v.reshape(B, L, Hk, Rk) @ b_v.reshape(B, L, Rk, Dh)) / Rk
        return (
            q,
            k,
            v,
            mx.sigmoid(s_k)[..., None],
            mx.sigmoid(s_v)[..., None],
            mx.sigmoid(lam)[..., None, None],
            gate.reshape(B, L, self.n_sig, Dh),
        )

    def __call__(self, h, kv_cache, shift_cache, mask, positions):
        """h: (B, L, D). positions: (B, L) absolute positions (0 = first token)."""
        B, L, _ = h.shape
        Hk, Dh = self.n_kv, self.head_dim
        q, k, v, alpha_k, alpha_v, lam, gate = self.project(h)

        # token shift: mix with the previous token's raw k / v
        if shift_cache is not None and shift_cache[0] is not None:
            k_last, v_last = (
                shift_cache[0].astype(mx.float32),
                shift_cache[1].astype(mx.float32),
            )
        else:
            k_last = mx.zeros((B, 1, Hk, Dh), dtype=mx.float32)
            v_last = mx.zeros((B, 1, Hk, Dh), dtype=mx.float32)
        k_prev = mx.concatenate([k_last, k[:, :-1]], axis=1)
        v_prev = mx.concatenate([v_last, v[:, :-1]], axis=1)
        if shift_cache is not None:
            lengths = getattr(shift_cache, "lengths", None)
            if lengths is not None and L > 1:
                idx = mx.clip(lengths - 1, 0, L - 1).reshape(B, 1, 1, 1)
                keep = (lengths > 0).reshape(B, 1, 1, 1)
                shift_cache[0] = mx.where(
                    keep, mx.take_along_axis(k, idx, axis=1), k_last
                )
                shift_cache[1] = mx.where(
                    keep, mx.take_along_axis(v, idx, axis=1), v_last
                )
            else:
                shift_cache[0] = k[:, -1:]
                shift_cache[1] = v[:, -1:]
            if hasattr(shift_cache, "advance"):
                shift_cache.advance(L)
        first = (positions <= 0)[..., None, None]  # document start: no shift
        alpha_k = mx.where(first, 0.0, alpha_k)
        alpha_v = mx.where(first, 0.0, alpha_v)
        k = alpha_k * k_prev + (1 - alpha_k) * k
        v = alpha_v * v_prev + (1 - alpha_v) * v

        q = mx.fast.rms_norm(
            q, _zc(self.q_norm.norm.weight, self.zc).astype(mx.float32), self.eps
        )
        k = mx.fast.rms_norm(
            k, _zc(self.k_norm.norm.weight, self.zc).astype(mx.float32), self.eps
        )
        # scalable softmax: q *= s_h * log(min(pos + 1, window))
        log_pos = mx.log(
            mx.minimum(positions.astype(mx.float32) + 1.0, float(self.window))
        )
        q = q * (self.softmax_scaler.astype(mx.float32) * log_pos[..., None])[..., None]

        dtype = h.dtype
        q = q.astype(dtype).transpose(0, 2, 1, 3)
        k = k.astype(dtype).transpose(0, 2, 1, 3)
        v = v.astype(dtype).transpose(0, 2, 1, 3)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        o = kernels.softcap_attention(
            q, k, v, self.scale, self.softcap, mask
        )  # (B,Hq,L,Dh)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_noise, self.snr + 1, Dh)
        o = o[..., : self.snr, :].astype(mx.float32) - lam * o[
            ..., self.snr :, :
        ].astype(mx.float32)
        o = o.reshape(B, L, self.n_sig, Dh) * nn.silu(gate + self.gate_bias)
        return o.reshape(B, L, -1)


def _cumsum(sizes):
    out, s = [], 0
    for n in sizes:
        s += n
        out.append(s)
    return out


class _Experts(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n: int):
        super().__init__()
        self.experts = SwitchLinear(d_in, d_hidden, n, bias=False)
        self.output_experts = SwitchLinear(d_hidden, d_in, n, bias=False)


class OlalaMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        d = args.hidden_size
        self.top_k = args.moe_num_active_experts
        self.n_experts = args.moe_num_routed_experts
        self.scaling = args.moe_routed_scaling_factor
        self.d_latent = args.moe_routed_input_dim
        self.d_shared = args.moe_shared_intermediate_size
        self.moe_gate = nn.Linear(d, self.n_experts, bias=False)
        self.expert_bias = mx.zeros((self.n_experts,))
        self.in_cat = nn.Linear(d, self.d_latent + self.d_shared, bias=False)
        self.out_cat = nn.Linear(self.d_latent + self.d_shared, d, bias=False)
        self.experts = _Experts(
            self.d_latent, args.moe_routed_intermediate_size, self.n_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.moe_gate(x.astype(self.moe_gate.weight.dtype)).astype(mx.float32)
        inds, probs = kernels.moe_route(
            logits, self.expert_bias, self.top_k, self.scaling
        )
        h = self.in_cat(x)
        x0, sh = h[..., : self.d_latent], h[..., self.d_latent :]
        e = kernels.moe_experts(x0, inds, probs.astype(x0.dtype), self.experts)
        return self.out_cat(mx.concatenate([e, mx.square(nn.relu(sh))], axis=-1))


class OlalaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        if layer_type == "M":
            self.mixer = Mamba3Mixer(args)
            mixer_out = self.mixer.d_inner
        else:
            self.mixer = DiffTPAttention(args, layer_idx)
            mixer_out = self.mixer.n_sig * self.mixer.head_dim
        self.mixer_proj = nn.Linear(mixer_out, args.hidden_size, bias=False)
        self.geodesic_mixer = _Geodesic()
        self.geodesic_mlp = _Geodesic()
        self.mlp = OlalaMoE(args)

    def __call__(
        self, x, dtype, cache=None, shift_cache=None, mask=None, positions=None
    ):
        h = x.astype(dtype)
        if self.layer_type == "M":
            y = self.mixer(h, cache, mask)
        else:
            y = self.mixer(h, cache, shift_cache, mask, positions)
        y = self.mixer_proj(y.astype(dtype))
        x = geodesic_update(
            x, y, self.geodesic_mixer.scale, self.geodesic_mixer.bias, self.layer_idx
        )
        y = self.mlp(x.astype(dtype))
        return geodesic_update(
            x, y, self.geodesic_mlp.scale, self.geodesic_mlp.bias, self.layer_idx
        )


class OlalaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [OlalaBlock(args, i, t) for i, t in enumerate(args.layers_config)]
        self.attn_idx = [i for i, t in enumerate(args.layers_config) if t == "V"]
        self.ssm_idx = [i for i, t in enumerate(args.layers_config) if t == "M"]

    def compute_dtype(self):
        dtype = self.embedding.weight.dtype
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            dtype = getattr(self.embedding, "scales", mx.zeros((1,), mx.bfloat16)).dtype
        return mx.float32 if _ACT_F32 else dtype

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(inputs)
        dtype = self.compute_dtype()
        x = inputs_embeds.astype(mx.float32)
        B, L, _ = x.shape

        if cache is None:
            cache = [None] * len(self.layers)

        kv0 = (
            cache[self.attn_idx[0]][0]
            if self.attn_idx and cache[self.attn_idx[0]] is not None
            else None
        )
        positions = _positions(kv0, B, L)
        c_ssm = cache[self.ssm_idx[0]] if self.ssm_idx else None
        ssm_mask = (
            c_ssm.make_mask(L)
            if c_ssm is not None and hasattr(c_ssm, "make_mask")
            else None
        )

        from . import metal

        if self.args.normalize_embeddings_ngpt and metal.can_decode(
            self, x, cache, ssm_mask
        ):
            return (
                metal.decode(self, x, cache, positions),
            )  # fused path, normalizes itself
        if self.args.normalize_embeddings_ngpt:
            x = x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)

        attn_mask = _attention_mask(kv0, L, self.args.slw_wsize)
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "M":
                x = layer(x, dtype, cache=cache[i], mask=ssm_mask)
            else:
                c = cache[i]
                x = layer(
                    x,
                    dtype,
                    cache=None if c is None else c[0],
                    shift_cache=None if c is None else c[1],
                    mask=attn_mask,
                    positions=positions,
                )
        return x


def _positions(kv_cache, B: int, L: int) -> mx.array:
    """Absolute positions (B, L) of the new tokens; left padding gives negatives."""
    if kv_cache is None:
        return mx.broadcast_to(mx.arange(L)[None], (B, L))
    offset = kv_cache.offset
    if isinstance(offset, int):
        return mx.broadcast_to((mx.arange(L) + offset)[None], (B, L))
    return offset[:, None] + mx.arange(L)[None]


def _attention_mask(kv_cache, L: int, window: int):
    if kv_cache is not None and hasattr(kv_cache, "make_mask"):
        offset = kv_cache.offset
        if isinstance(offset, int) and offset + L <= window:
            if L == 1:
                return None
            if offset == 0:
                return "causal"
        try:
            return kv_cache.make_mask(L, window_size=window, return_array=True)
        except TypeError:
            return kv_cache.make_mask(L, return_array=True)
    if L == 1:
        return None
    return "causal"


def quant_predicate(path: str, module) -> bool:
    """Keep the router and the small per-head projections in full precision."""
    return not path.endswith(("moe_gate", "in_proj_small"))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OlalaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        def head_fn(x):
            from . import metal

            return metal.head(self.lm_head, x)

        # plain function attribute (not a module): lets the fused path warm up the head kernels
        self.model._head_fn = head_fn

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None, **kwargs):
        x = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        if isinstance(x, tuple):  # fused decode: f32 residual, head on f32 activations
            from . import metal

            return metal.head(self.lm_head, x[0])
        return self.lm_head(x.astype(self.model.compute_dtype()))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> list[Any]:
        return [
            ArraysCache(1)
            if layer.layer_type == "M"
            else CacheList(KVCache(), ArraysCache(2))
            for layer in self.model.layers
        ]

    def sanitize(self, weights: dict) -> dict:
        """Map a HF checkpoint (or an already converted MLX one) to this module tree."""
        w = {k: v for k, v in weights.items() if not k.endswith(".prosres_scalar")}
        for k in list(w):
            if (".geodesic_mixer." in k or ".geodesic_mlp." in k) and w[k].ndim == 0:
                w[k] = w[k].reshape(1)

        def cat(dst, srcs, axis=0):
            # merge weight matrices (and their quantization scales / biases, if any)
            for suffix in ("weight", "scales", "biases"):
                parts = [f"{s}.{suffix}" for s in srcs]
                if all(p in w for p in parts):
                    w[f"{dst}.{suffix}"] = mx.concatenate(
                        [w.pop(p) for p in parts], axis=axis
                    )

        for i, t in enumerate(self.args.layers_config):
            p = f"model.layers.{i}"
            if t == "M" and f"{p}.mixer.in_proj_dyn.weight" in w:
                cat(
                    f"{p}.mixer.in_proj",
                    [f"{p}.mixer.in_proj", f"{p}.mixer.in_proj_dyn"],
                )
            if t == "V" and f"{p}.mixer.c_q.weight" in w:
                m = f"{p}.mixer"
                cat(
                    f"{m}.in_proj",
                    [f"{m}.c_q", f"{m}.W_B_k", f"{m}.W_B_v", f"{p}.gate_proj"],
                )
                cat(
                    f"{m}.in_proj_small",
                    [
                        f"{m}.W_A_k",
                        f"{m}.W_A_v",
                        f"{m}.shift_proj_k",
                        f"{m}.shift_proj_v",
                        f"{m}.lambda_proj",
                    ],
                )
            if f"{p}.mlp.down_proj.weight" in w:
                m = f"{p}.mlp"
                cat(f"{m}.in_cat", [f"{m}.down_proj", f"{m}.shared_experts.fc_1"])
                cat(
                    f"{m}.out_cat", [f"{m}.up_proj", f"{m}.shared_experts.fc_2"], axis=1
                )
        if self.args.tie_word_embeddings:
            w.pop("lm_head.weight", None)
        return w

    @property
    def quant_predicate(self):
        return quant_predicate

    @property
    def cast_predicate(self):
        big = (
            "in_proj.weight",
            "in_proj_small.weight",
            "mixer_proj.weight",
            "in_cat.weight",
            "out_cat.weight",
            "experts.weight",
            "output_experts.weight",
            "embedding.weight",
            "lm_head.weight",
        )

        def predicate(path: str):
            # only the large matrices follow --dtype; the router, the per-head parameters and
            # the geodesic scalars keep the checkpoint's f32
            return path.endswith(big)

        return predicate
