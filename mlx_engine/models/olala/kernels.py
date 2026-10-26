"""Compute kernels for Olala.

Reference implementations in plain MLX ops; fused Metal kernels are used for
the hot paths when available (see metal.py).
"""

import math

import mlx.core as mx
from mlx import nn
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

# chunk length (tokens) of the chunked Mamba-3 prefill
M3_CHUNK = 32


# --------------------------------------------------------------------------
# Mamba-3 MIMO
# --------------------------------------------------------------------------


def _m3_prep(proj, dyn, m, mask):
    """Split the projections and discretize. Returns f32 tensors:
    x, z (B,L,H,P); dt, la (log alpha), trap (B,L,H); Bn, Cn (B,L,R,N); ang_inc (B,L,H,A)."""
    Bsz, L, _ = proj.shape
    H, R, N, P, _A = m.H, m.R, m.N, m.P, m.A
    ph = proj.reshape(Bsz, L, H, 2 * P + 3).astype(mx.float32)
    z = ph[..., :P]
    x = ph[..., P : 2 * P]
    dt = nn.softplus(ph[..., 2 * P] + m.dt_bias.astype(mx.float32))
    a = mx.minimum(-nn.softplus(ph[..., 2 * P + 1]), -1e-4)
    trap = mx.sigmoid(ph[..., 2 * P + 2])
    if mask is not None:
        valid = mask.astype(mx.bool_)[..., None]  # (B, L, 1)
        dt = mx.where(valid, dt, 0.0)
        x = mx.where(valid[..., None], x, 0.0)
    dyn = dyn.astype(mx.float32)
    bc = dyn[..., : 2 * R * N].reshape(Bsz, L, 2 * R, N)
    wb = (
        (1.0 + m.B_norm.norm.weight.astype(mx.float32))
        if m.zc
        else m.B_norm.norm.weight.astype(mx.float32)
    )
    wc = (
        (1.0 + m.C_norm.norm.weight.astype(mx.float32))
        if m.zc
        else m.C_norm.norm.weight.astype(mx.float32)
    )
    bc = bc * mx.rsqrt(mx.mean(bc * bc, axis=-1, keepdims=True) + m.eps)
    Bn = bc[:, :, :R] * wb
    Cn = bc[:, :, R:] * wc
    ang = mx.tanh(dyn[..., 2 * R * N :]) * math.pi  # (B, L, A)
    ang_inc = ang[:, :, None, :] * dt[..., None]  # (B, L, H, A)
    return x, z, dt, a * dt, trap, Bn, Cn, ang_inc


def m3_unpack(st, m):
    """Packed state (B, H, P*N + R*N + P + A) -> (angle, S, k, x)."""
    B = st.shape[0]
    H, R, N, P, _A = m.H, m.R, m.N, m.P, m.A
    o1, o2, o3 = P * N, P * N + R * N, P * N + R * N + P
    return (
        st[..., o3:],
        st[..., :o1].reshape(B, H, P, N),
        st[..., o1:o2].reshape(B, H, R, N),
        st[..., o2:o3],
    )


def m3_pack(state, m):
    ang, S, k, x = state
    B, H = S.shape[:2]
    return mx.concatenate(
        [
            S.reshape(B, H, -1),
            k.reshape(B, H, -1).astype(mx.float32),
            x.astype(mx.float32),
            ang.astype(mx.float32),
        ],
        axis=-1,
    )


def _rotate(t, cos, sin, A):
    """Rotate pairs (i, i + 2A) for i < A along the last axis. t: (..., N); cos/sin: (..., A)."""
    t0 = t[..., :A]
    t2 = t[..., 2 * A : 3 * A]
    r0 = t0 * cos - t2 * sin
    r2 = t0 * sin + t2 * cos
    return mx.concatenate([r0, t[..., A : 2 * A], r2, t[..., 3 * A :]], axis=-1)


def _wrap(theta):
    two_pi = 2.0 * math.pi
    return theta - two_pi * mx.round(theta / two_pi)


def mamba3_forward(proj, dyn, state, m, mask=None, lengths=None):
    """Mamba-3 MIMO mixer core.

    proj: (B, L, H*(2P+3)) in_proj output; dyn: (B, L, 2RN + A) in_proj_dyn output.
    state: None or (angle (B,H,A), ssm (B,H,P,N), k (B,H,R,N), x (B,H,P)).
    Returns y (B, L, H*P) in proj.dtype and the new state.
    """
    Bsz, L, _ = proj.shape
    H, R, N, P, A = m.H, m.R, m.N, m.P, m.A
    if state is None:
        state = (
            mx.zeros((Bsz, H, A), mx.float32),
            mx.zeros((Bsz, H, P, N), mx.float32),
            mx.zeros((Bsz, H, R, N), mx.float32),
            mx.zeros((Bsz, H, P), mx.float32),
        )
    if L == 1 and mask is None:
        return _mamba3_step_ref(proj, dyn, state, m)
    return _mamba3_chunked_ref(proj, dyn, state, m, mask, lengths)


def _mamba3_step_ref(proj, dyn, state, m):
    Bsz = proj.shape[0]
    H, _R, _N, P, A = m.H, m.R, m.N, m.P, m.A
    ang0, S, k_prev, x_prev = state
    x, z, dt, la, trap, Bn, Cn, ang_inc = _m3_prep(proj, dyn, m, None)
    x, z, dt, la, trap = (
        x[:, 0],
        z[:, 0],
        dt[:, 0],
        la[:, 0],
        trap[:, 0],
    )  # (B,H,P) / (B,H)
    theta = _wrap(ang0 + ang_inc[:, 0])  # (B,H,A)
    cos, sin = mx.cos(theta)[:, :, None], mx.sin(theta)[:, :, None]
    k = Bn[:, 0][:, None] + m.B_bias.astype(mx.float32)  # (B,H,R,N)
    q = Cn[:, 0][:, None] + m.C_bias.astype(mx.float32)
    k = _rotate(k, cos, sin, A)
    q = _rotate(q, cos, sin, A)
    mx_ = m.in_proj_mimo_x.astype(mx.float32)  # (H,R,P)
    v = x[:, :, None, :] * mx_  # (B,H,R,P)
    v_prev = x_prev[:, :, None, :] * mx_
    w = nn.silu(
        z[:, :, None, :] * m.in_proj_mimo_z.astype(mx.float32)
    ) * m.out_proj_mimo.astype(mx.float32)
    alpha = mx.exp(la)
    gamma = trap * dt
    beta = (1.0 - trap) * dt * alpha
    u = v.swapaxes(-1, -2) @ k  # (B,H,P,N)
    u_prev = v_prev.swapaxes(-1, -2) @ k_prev.astype(mx.float32)
    S = (
        alpha[..., None, None] * S
        + beta[..., None, None] * u_prev
        + gamma[..., None, None] * u
    )
    sq = (S @ q.swapaxes(-1, -2)).swapaxes(-1, -2)  # (B,H,R,P)
    y = mx.sum(
        w * (m.D.astype(mx.float32)[None, :, None, None] * v + sq), axis=2
    )  # (B,H,P)
    y = y.reshape(Bsz, 1, H * P).astype(proj.dtype)
    return y, (theta, S, k, x)


def _mamba3_chunked_ref(proj, dyn, state, m, mask, lengths):
    Bsz, L, _ = proj.shape
    H, R, N, P, A = m.H, m.R, m.N, m.P, m.A
    T = M3_CHUNK
    ang0, S, k_last, x_last = state
    x, z, dt, la, trap, Bn, Cn, ang_inc = _m3_prep(proj, dyn, m, mask)

    # cumulative rotary angles (wrapped)
    theta = _wrap(ang0[:, None] + mx.cumsum(ang_inc, axis=1))  # (B,L,H,A)
    cos, sin = mx.cos(theta)[:, :, :, None], mx.sin(theta)[:, :, :, None]
    k = _rotate(
        Bn[:, :, None] + m.B_bias.astype(mx.float32), cos, sin, A
    )  # (B,L,H,R,N)
    q = _rotate(Cn[:, :, None] + m.C_bias.astype(mx.float32), cos, sin, A)
    mx_ = m.in_proj_mimo_x.astype(mx.float32)
    v = x[:, :, :, None, :] * mx_  # (B,L,H,R,P)
    w = nn.silu(
        z[:, :, :, None, :] * m.in_proj_mimo_z.astype(mx.float32)
    ) * m.out_proj_mimo.astype(mx.float32)
    gamma = trap * dt  # (B,L,H)
    delta = (1.0 - trap) * dt

    # pad to a multiple of T (padding: dt = 0, x = 0 -> no effect)
    Lp = ((L + T - 1) // T) * T
    pad = Lp - L

    def padL(t):
        if pad == 0:
            return t
        widths = [(0, 0)] * t.ndim
        widths[1] = (0, pad)
        return mx.pad(t, widths)

    k, q, v, la, gamma, delta = map(padL, (k, q, v, la, gamma, delta))
    Nc = Lp // T
    # (B, H, Nc, T, ...)
    k = k.reshape(Bsz, Nc, T, H, R, N).transpose(0, 3, 1, 2, 4, 5)
    q = q.reshape(Bsz, Nc, T, H, R, N).transpose(0, 3, 1, 2, 4, 5)
    v = v.reshape(Bsz, Nc, T, H, R, P).transpose(0, 3, 1, 2, 4, 5)
    la = la.reshape(Bsz, Nc, T, H).transpose(0, 3, 1, 2)
    gamma = gamma.reshape(Bsz, Nc, T, H).transpose(0, 3, 1, 2)
    delta = delta.reshape(Bsz, Nc, T, H).transpose(0, 3, 1, 2)

    cum = mx.cumsum(la, axis=-1)  # (B,H,Nc,T)
    delta_next = mx.concatenate(
        [delta[..., 1:], mx.zeros_like(delta[..., :1])], axis=-1
    )
    g = gamma + delta_next  # weight of u_s for later tokens of the chunk
    seg = cum[..., :, None] - cum[..., None, :]  # (B,H,Nc,T,T) t, s
    tri = mx.tril(mx.ones((T, T), dtype=mx.bool_), k=-1)
    eye = mx.eye(T, dtype=mx.bool_)
    Mx = mx.where(tri, mx.exp(mx.where(tri, seg, 0.0)) * g[..., None, :], 0.0)
    Mx = Mx + mx.where(eye, gamma[..., :, None], 0.0)

    kf = k.reshape(Bsz, H, Nc, T * R, N)
    qf = q.reshape(Bsz, H, Nc, T * R, N)
    vf = v.reshape(Bsz, H, Nc, T * R, P)
    scores = (qf @ kf.swapaxes(-1, -2)).reshape(Bsz, H, Nc, T, R, T, R)
    scores = scores * Mx[:, :, :, :, None, :, None]
    intra = scores.reshape(Bsz, H, Nc, T * R, T * R) @ vf  # (B,H,Nc,TR,P)

    # per-chunk state increments
    e = mx.exp(cum[..., -1:] - cum) * g  # (B,H,Nc,T)
    e = mx.concatenate([e[..., :-1], gamma[..., -1:]], axis=-1)
    ve = (v * e[..., None, None]).reshape(Bsz, H, Nc, T * R, P)
    dS = ve.swapaxes(-1, -2) @ kf  # (B,H,Nc,P,N)
    a_c = mx.exp(cum[..., -1])  # (B,H,Nc)
    # last-token outer product of each chunk (feeds the next chunk's trapezoid term)
    u_last = v[:, :, :, -1].swapaxes(-1, -2) @ k[:, :, :, -1]  # (B,H,Nc,P,N)
    v_prev0 = x_last[:, :, None, :] * mx_  # (B,H,R,P)
    u_prev0 = v_prev0.swapaxes(-1, -2) @ k_last.astype(mx.float32)  # (B,H,P,N)

    S_inits = []
    for c in range(Nc):
        u_prev = u_prev0 if c == 0 else u_last[:, :, c - 1]
        S_eff = S + delta[:, :, c, 0][..., None, None] * u_prev
        S_inits.append(S_eff)
        S = a_c[:, :, c][..., None, None] * S_eff + dS[:, :, c]
    S_init = mx.stack(S_inits, axis=2)  # (B,H,Nc,P,N)
    inter = (qf @ S_init.swapaxes(-1, -2)) * mx.repeat(mx.exp(cum), R, axis=-1)[
        ..., None
    ]
    out = (intra + inter).reshape(Bsz, H, Nc, T, R, P)
    D = m.D.astype(mx.float32)[None, :, None, None, None, None]
    w = padL(w).reshape(Bsz, Nc, T, H, R, P).transpose(0, 3, 1, 2, 4, 5)
    y = mx.sum(w * (D * v + out), axis=4)  # (B,H,Nc,T,P)
    y = y.transpose(0, 2, 3, 1, 4).reshape(Bsz, Lp, H * P)[:, :L].astype(proj.dtype)

    # new state: last valid token
    k_all = k.transpose(0, 2, 3, 1, 4, 5).reshape(Bsz, Lp, H, R, N)
    if lengths is not None:
        idx = mx.clip(lengths - 1, 0, L - 1)
        has = (lengths > 0).reshape(Bsz, 1, 1)
        last_k = mx.take_along_axis(k_all, idx.reshape(Bsz, 1, 1, 1, 1), axis=1)[:, 0]
        last_x = mx.take_along_axis(x, idx.reshape(Bsz, 1, 1, 1), axis=1)[:, 0]
        last_t = mx.take_along_axis(theta, idx.reshape(Bsz, 1, 1, 1), axis=1)[:, 0]
        new_k = mx.where(has[..., None], last_k, k_last.astype(mx.float32))
        new_x = mx.where(has, last_x, x_last.astype(mx.float32))
        new_theta = mx.where(has, last_t, ang0)
    else:
        new_k = k_all[:, L - 1]
        new_x = x[:, L - 1]
        new_theta = theta[:, L - 1]
    return y, (new_theta, S, new_k, new_x)


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------


ATTN_SCORE_BUDGET = 512 * 1024 * 1024  # bytes of f32 scores materialized at once


def softcap_attention(q, k, v, scale, softcap, mask):
    """q: (B,Hq,L,D), k/v: (B,Hk,T,D). mask: None, "causal" or bool array broadcastable to (B,1,L,T).
    Queries are processed in chunks so that the f32 score matrix stays under ATTN_SCORE_BUDGET."""
    B, Hq, L, _D = q.shape
    T = k.shape[2]
    qc = max(1, ATTN_SCORE_BUDGET // (4 * B * Hq * T))
    if qc >= L:
        return _softcap_attention_chunk(q, k, v, scale, softcap, mask, 0)
    outs = []
    for s0 in range(0, L, qc):
        s1 = min(L, s0 + qc)
        m = mask
        if m is not None and not isinstance(m, str):
            m = m[..., s0:s1, :] if m.shape[-2] == L else m
        outs.append(
            _softcap_attention_chunk(q[:, :, s0:s1], k, v, scale, softcap, m, s0, L)
        )
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=2)


def _softcap_attention_chunk(q, k, v, scale, softcap, mask, q0, L_total=None):
    B, Hq, L, D = q.shape
    Hk, T = k.shape[1], k.shape[2]
    L_total = L_total or L
    G = Hq // Hk
    qg = q.reshape(B, Hk, G, L, D)
    s = (qg @ k[:, :, None].swapaxes(-1, -2)).astype(mx.float32) * scale  # (B,Hk,G,L,T)
    s = mx.tanh(s / softcap) * softcap
    if isinstance(mask, str):
        qi = mx.arange(T - L_total + q0, T - L_total + q0 + L)[:, None]
        ki = mx.arange(T)[None]
        s = mx.where(ki <= qi, s, -mx.inf)
    elif mask is not None:
        mk = mask
        if mk.ndim == 4:  # (B or 1, 1, L, T)
            mk = mk[:, :, None]
        s = mx.where(mk, s, -mx.inf)
    p = mx.softmax(s, axis=-1, precise=True).astype(v.dtype)
    o = p @ v[:, :, None]
    return o.reshape(B, Hq, L, D)


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------


def moe_route(logits, expert_bias, top_k, scaling):
    """Sigmoid router with selection bias. Returns (indices (...,K) uint32, probs (...,K) f32)."""
    scores = mx.sigmoid(logits)
    sel = scores + expert_bias.astype(mx.float32)
    inds = mx.argpartition(-sel, kth=top_k - 1, axis=-1)[..., :top_k]
    probs = mx.take_along_axis(scores, inds, axis=-1)
    probs = probs / (mx.sum(probs, axis=-1, keepdims=True) + 1e-20) * scaling
    return inds.astype(mx.uint32), probs


def moe_experts(x, inds, probs, experts):
    """x: (..., d_in); inds/probs: (..., K). Returns sum_k probs_k * expert_k(x)."""
    xe = mx.expand_dims(x, (-2, -3))
    do_sort = inds.size >= 64
    idx = inds
    if do_sort:
        xe, idx, inv_order = _gather_sort(xe, inds)
    h = experts.experts(xe, idx, sorted_indices=do_sort)
    h = mx.square(nn.relu(h))
    o = experts.output_experts(h, idx, sorted_indices=do_sort)
    if do_sort:
        o = _scatter_unsort(o, inv_order, inds.shape)
    o = o.squeeze(-2)  # (..., K, d_in)
    return (o * probs[..., None]).sum(axis=-2).astype(x.dtype)
