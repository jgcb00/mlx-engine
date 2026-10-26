"""Olala 7A1B (OlalaForCausalLM) for mlx-lm / mlx-vlm.

Hybrid model: 30 Mamba-3 MIMO layers ("M") and 6 differential tensor-product
attention layers ("V"), a 256-expert MoE (6 active, latent 384) on every
layer, and a geodesic (rotation) residual update instead of pre-norms.

Reference: modeling_olala.py shipped with the checkpoints, and the llama.cpp
implementation (src/models/olala.cpp) it was validated against.

Cache layout (one entry per item of ``make_cache()``):
  - layer i of type M: ArraysCache(4) = [angle (B,H,A), ssm (B,H,P,N) f32,
    k (B,H,R,N) rotated keys of the last token, x (B,H,P) raw x of the last token]
  - layer i of type V: KVCache (keys after token shift and k-norm)
  - then one ArraysCache(2) per V layer = [k_raw, v_raw] of the last token
    (token shift input)
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.switch_layers import SwitchLinear

from . import kernels

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


def _zc(weight: mx.array, zero_centered: bool) -> mx.array:
    return 1.0 + weight if zero_centered else weight


class _Norm(nn.Module):
    """RMSNorm stored as ``<name>.norm.weight`` (checkpoint layout)."""

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
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2 + 3 * H, bias=False)
        self.in_proj_dyn = nn.Linear(self.d_model, 2 * R * N + self.A, bias=False)
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
        Bsz, L, _ = h.shape
        H, R, N, P, A = self.H, self.R, self.N, self.P, self.A
        proj = self.in_proj(h)
        dyn = self.in_proj_dyn(h)

        if cache is not None and cache[0] is not None:
            state = (cache[0], cache[1], cache[2], cache[3])
        else:
            state = None

        y, new_state = kernels.mamba3_forward(
            proj,
            dyn,
            state,
            self,
            mask,
            lengths=None if cache is None else getattr(cache, "lengths", None),
        )
        if cache is not None:
            cache[0], cache[1], cache[2], cache[3] = new_state
            if hasattr(cache, "advance"):
                cache.advance(L)
        return y


class DiffTPAttention(nn.Module):
    """Differential attention V2 with tensor-product K/V (rank 4) and token shift."""

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
        self.scale = (
            1.0 / self.head_dim if args.use_completed_p else self.head_dim**-0.5
        )

        self.c_q = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.W_A_k = nn.Linear(d, self.n_kv * self.rank, bias=False)
        self.W_A_v = nn.Linear(d, self.n_kv * self.rank, bias=False)
        self.W_B_k = nn.Linear(d, self.rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(d, self.rank * self.head_dim, bias=False)
        self.shift_proj_k = nn.Linear(d, self.n_kv, bias=False)
        self.shift_proj_v = nn.Linear(d, self.n_kv, bias=False)
        self.q_norm = _Norm(self.head_dim)
        self.k_norm = _Norm(self.head_dim)
        self.softmax_scaler = mx.ones((self.n_heads,))
        self.lambda_proj = nn.Linear(d, self.n_noise, bias=False)

    def __call__(self, h, kv_cache, shift_cache, mask, positions):
        """h: (B, L, D). positions: (B, L) int32 absolute positions (0 = first token)."""
        B, L, _ = h.shape
        Hq, Hk, Dh, Rk = self.n_heads, self.n_kv, self.head_dim, self.rank

        q = self.c_q(h).reshape(B, L, Hq, Dh)
        a_k = self.W_A_k(h).reshape(B, L, Hk, Rk)
        a_v = self.W_A_v(h).reshape(B, L, Hk, Rk)
        b_k = self.W_B_k(h).reshape(B, L, Rk, Dh)
        b_v = self.W_B_v(h).reshape(B, L, Rk, Dh)
        k = (a_k @ b_k) / Rk  # (B, L, Hk, Dh)
        v = (a_v @ b_v) / Rk

        # token shift: mix with the previous token's raw k / v
        alpha_k = mx.sigmoid(self.shift_proj_k(h).astype(mx.float32))[..., None]
        alpha_v = mx.sigmoid(self.shift_proj_v(h).astype(mx.float32))[..., None]
        if shift_cache is not None and shift_cache[0] is not None:
            k_last, v_last = shift_cache[0], shift_cache[1]
        else:
            k_last = mx.zeros((B, 1, Hk, Dh), dtype=k.dtype)
            v_last = mx.zeros((B, 1, Hk, Dh), dtype=v.dtype)
        k_prev = mx.concatenate([k_last.astype(k.dtype), k[:, :-1]], axis=1)
        v_prev = mx.concatenate([v_last.astype(v.dtype), v[:, :-1]], axis=1)
        if shift_cache is not None:
            lengths = getattr(shift_cache, "lengths", None)
            if lengths is not None and L > 1:
                idx = mx.clip(lengths - 1, 0, L - 1).reshape(B, 1, 1, 1)
                new_k = mx.take_along_axis(k, idx, axis=1)
                new_v = mx.take_along_axis(v, idx, axis=1)
                keep = (lengths > 0).reshape(B, 1, 1, 1)
                shift_cache[0] = mx.where(keep, new_k, k_last.astype(k.dtype))
                shift_cache[1] = mx.where(keep, new_v, v_last.astype(v.dtype))
            else:
                shift_cache[0] = k[:, -1:]
                shift_cache[1] = v[:, -1:]
            if hasattr(shift_cache, "advance"):
                shift_cache.advance(L)
        first = (positions <= 0)[..., None, None]  # document start: no shift
        alpha_k = mx.where(first, 0.0, alpha_k)
        alpha_v = mx.where(first, 0.0, alpha_v)
        k = (alpha_k * k_prev + (1 - alpha_k) * k).astype(h.dtype)
        v = (alpha_v * v_prev + (1 - alpha_v) * v).astype(h.dtype)

        q = mx.fast.rms_norm(
            q, _zc(self.q_norm.norm.weight, self.zc).astype(q.dtype), self.eps
        )
        k = mx.fast.rms_norm(
            k, _zc(self.k_norm.norm.weight, self.zc).astype(k.dtype), self.eps
        )

        # scalable softmax: q *= s_h * log(min(pos + 1, window))
        log_pos = mx.log(
            mx.minimum(positions.astype(mx.float32) + 1.0, float(self.window))
        )
        q = (
            q.astype(mx.float32) * (self.softmax_scaler * log_pos[..., None])[..., None]
        ).astype(h.dtype)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)

        o = kernels.softcap_attention(
            q, k, v, self.scale, self.softcap, mask
        )  # (B, Hq, L, Dh)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_noise, self.snr + 1, Dh)
        lam = mx.sigmoid(self.lambda_proj(h).astype(mx.float32))[..., None, None]
        o = o[..., : self.snr, :].astype(mx.float32) - lam * o[
            ..., self.snr :, :
        ].astype(mx.float32)
        return o.reshape(B, L, self.n_sig, Dh)


class SharedMLP(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.fc_1 = nn.Linear(d, inter, bias=False)
        self.fc_2 = nn.Linear(inter, d, bias=False)

    def __call__(self, x):
        return self.fc_2(mx.square(nn.relu(self.fc_1(x))))


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
        self.moe_gate = nn.Linear(d, self.n_experts, bias=False)
        self.expert_bias = mx.zeros((self.n_experts,))
        self.down_proj = nn.Linear(d, args.moe_routed_input_dim, bias=False)
        self.up_proj = nn.Linear(args.moe_routed_input_dim, d, bias=False)
        self.experts = _Experts(
            args.moe_routed_input_dim, args.moe_routed_intermediate_size, self.n_experts
        )
        self.shared_experts = SharedMLP(d, args.moe_shared_intermediate_size)

    def route(self, x: mx.array):
        logits = self.moe_gate(x.astype(self.moe_gate.weight.dtype)).astype(mx.float32)
        return kernels.moe_route(logits, self.expert_bias, self.top_k, self.scaling)

    def __call__(self, x: mx.array) -> mx.array:
        inds, probs = self.route(x)
        x0 = self.down_proj(x)
        out = kernels.moe_experts(x0, inds, probs.astype(x0.dtype), self.experts)
        return self.up_proj(out) + self.shared_experts(x)


class OlalaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        if layer_type == "M":
            self.mixer = Mamba3Mixer(args)
            mixer_out = self.mixer.d_inner
            self.use_gate = False
        else:
            self.mixer = DiffTPAttention(args, layer_idx)
            mixer_out = self.mixer.n_sig * self.mixer.head_dim
            self.use_gate = args.gate_attn
            if self.use_gate:
                self.gate_proj = nn.Linear(args.hidden_size, mixer_out, bias=False)
                self.gate_bias = 1.15 if args.zero_centered_gate else 0.0
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
            if self.use_gate:
                g = self.gate_proj(h).astype(mx.float32).reshape(y.shape)
                y = y * nn.silu(g + self.gate_bias)
            y = y.reshape(*y.shape[:2], -1)
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

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(inputs)
        dtype = self.embedding.weight.dtype
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            dtype = mx.bfloat16
        if _ACT_F32:
            dtype = mx.float32
        x = inputs_embeds.astype(mx.float32)
        if self.args.normalize_embeddings_ngpt:
            x = x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)
        B, L, _ = x.shape

        n_layers = len(self.layers)
        if cache is None:
            cache = [None] * (n_layers + len(self.attn_idx))
        shift = dict(zip(self.attn_idx, cache[n_layers:]))

        kv0 = cache[self.attn_idx[0]] if self.attn_idx else None
        positions = _positions(kv0, B, L)
        attn_mask = _attention_mask(kv0, L, self.args.slw_wsize)
        ssm_mask = None
        c_ssm = cache[self.ssm_idx[0]] if self.ssm_idx else None
        if c_ssm is not None and hasattr(c_ssm, "make_mask"):
            ssm_mask = c_ssm.make_mask(L)

        for i, layer in enumerate(self.layers):
            if layer.layer_type == "M":
                x = layer(x, dtype, cache=cache[i], mask=ssm_mask)
            else:
                x = layer(
                    x,
                    dtype,
                    cache=cache[i],
                    shift_cache=shift[i],
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
        total = (offset if isinstance(offset, int) else 0) + L
        if isinstance(offset, int) and total <= window:
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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OlalaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None, **kwargs):
        x = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return self.lm_head(
            x.astype(
                self.lm_head.weight.dtype
                if self.lm_head.weight.dtype != mx.uint32
                else mx.bfloat16
            )
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = []
        for layer in self.model.layers:
            caches.append(ArraysCache(4) if layer.layer_type == "M" else KVCache())
        caches.extend(ArraysCache(2) for _ in self.model.attn_idx)
        return caches

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.endswith(".prosres_scalar"):
                continue
            if (".geodesic_mixer." in k or ".geodesic_mlp." in k) and v.ndim == 0:
                v = v.reshape(1)
            out[k] = v
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        return out

    @property
    def quant_predicate(self):
        def predicate(path, module):
            # keep the router, the tiny per-head projections and the norms in full precision
            if (
                path.endswith("moe_gate")
                or path.endswith("shift_proj_k")
                or path.endswith("shift_proj_v")
            ):
                return False
            if (
                path.endswith("lambda_proj")
                or path.endswith("W_A_k")
                or path.endswith("W_A_v")
            ):
                return False
            return True

        return predicate
