"""Fused Metal kernels for Olala decoding (mx.fast.metal_kernel).

One decode step (L == 1, batch B <= MAX_ROWS) runs as a short chain of fused
kernels per layer instead of the ~60 small ops of the reference path:

  M layer: gemv[geo](in_proj) -> m3_step -> gemv(mixer_proj)
           -> gemv[geo](router | in_cat) -> moe_up[top-k] -> moe_down -> gemv[relu2cat](out_cat)
  V layer: gemv[geo](in_proj, in_proj_small) -> v_prep -> KV append -> attn -> attn_combine
           -> gemv(mixer_proj) -> MoE as above

Activations stay in f32 (the GEMV kernels read bf16 / f16 / affine-quantized
weights and accumulate in f32). The geodesic residual update of the previous
sub-layer runs as the prologue of the next matrix-vector product ("geo").
"""

import math
import os

import mlx.core as mx
from mlx import nn

MAX_ROWS = 8
NSG = int(os.environ.get("OLALA_NSG", "16"))  # simdgroups per threadgroup in the GEMV kernels
TG_X_BYTES = 28 * 1024  # threadgroup memory budget for a computed GEMV input

_HDR = """
#include <metal_simdgroup>
using namespace metal;

inline float olala_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-fabs(x))); }
inline float olala_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float4 relu2(float4 v) { v = max(v, 0.0f); return v * v; }
"""


def enabled() -> bool:
    return os.environ.get("OLALA_MLX_REF", "0") != "1" and mx.metal.is_available()


# --------------------------------------------------------------------------
# weight handles
# --------------------------------------------------------------------------


class W:
    """A (rows, cols) weight in one of the supported formats."""

    def __init__(self, module_or_array, scales=None, biases=None, bits=None, group_size=None):
        self._arrays = None
        if isinstance(module_or_array, mx.array):
            self.w, self.scales, self.biases = module_or_array, scales, biases
            self.bits, self.gs = bits, group_size
        else:
            m = module_or_array
            self.w = m.weight
            self.scales = getattr(m, "scales", None)
            self.biases = getattr(m, "biases", None)
            self.bits = getattr(m, "bits", None)
            self.gs = getattr(m, "group_size", None)
            if getattr(m, "mode", "affine") != "affine":
                raise ValueError("unsupported quantization mode")
        if self.scales is not None:
            self.fmt = f"q{self.bits}"
            if self.bits not in (4, 8) or self.gs not in (32, 64, 128):
                raise ValueError("unsupported quantization")
            self.rows = self.w.shape[-2]
            self.cols = self.w.shape[-1] * 32 // self.bits
        else:
            self.fmt = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[self.w.dtype]
            self.rows, self.cols = self.w.shape[-2], self.w.shape[-1]

    def arrays(self):
        if self._arrays is None:
            self._arrays = [_handle(a) for a in ([self.w] if self.scales is None else [self.w, self.scales, self.biases])]
        return self._arrays

    def key(self):
        return f"{self.fmt}{self.gs or ''}"


def _handle(a: mx.array) -> mx.array:
    """A 16-element view of a (contiguous) weight array.

    The kernels index the whole underlying buffer through it. MLX commits a
    command buffer once the bytes of the arrays bound in it exceed
    MLX_MAX_MB_PER_BUFFER; binding the full 75 MB expert tensors or the 470 MB
    head forced a commit almost every kernel (measured: expert GEMV 28 us ->
    10 us). Weights are read-only, so the view changes no dependency."""
    return a.reshape(-1)[:16]


def _wtype(fmt):
    return {"bf16": "bfloat", "f16": "half", "f32": "float"}.get(fmt, "uint32_t")


def _chunk(fmt):
    # values per 16-byte load
    return {"bf16": 8, "f16": 8, "f32": 4, "q8": 16, "q4": 32}[fmt]


def _dot_code(fmt, name, K, gs, xexpr, tdim, rowvar, kvar):
    """Code accumulating acc[r][t] += dot(W[row, k:k+chunk], x[t, k:k+chunk]) for one 16-byte chunk.
    xexpr(t, j) must give a float4 of x for token t at k + 4*j."""
    lines = []
    if fmt in ("bf16", "f16"):
        vt = "bfloat4" if fmt == "bf16" else "half4"
        lines.append(f"const device {vt}* wp = (const device {vt}*)({name} + (size_t){rowvar} * {K} + {kvar});")
        lines.append("const float4 wv0 = float4(wp[0]), wv1 = float4(wp[1]);")
        lines.append(f"for (int t = 0; t < {tdim}; ++t) acc[r][t] += dot(wv0, {xexpr('t', 0)}) + dot(wv1, {xexpr('t', 1)});")
    elif fmt == "f32":
        lines.append(f"const float4 wv0 = *(const device float4*)({name} + (size_t){rowvar} * {K} + {kvar});")
        lines.append(f"for (int t = 0; t < {tdim}; ++t) acc[r][t] += dot(wv0, {xexpr('t', 0)});")
    elif fmt == "q8":
        lines.append(f"const uint4 q = *(const device uint4*)({name} + (size_t){rowvar} * ({K} / 4) + {kvar} / 4);")
        lines.append(f"const size_t gi = (size_t){rowvar} * ({K} / {gs}) + {kvar} / {gs};")
        lines.append(f"const float sc = float({name}_s[gi]), bi = float({name}_b[gi]);")
        lines.append("float4 f[4];")
        lines.append("for (int j = 0; j < 4; ++j) { const uint u = q[j]; f[j] = float4(u & 0xffu, (u >> 8) & 0xffu, (u >> 16) & 0xffu, u >> 24); }")
        lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0, s2 = 0;")
        lines.append(f"  for (int j = 0; j < 4; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}")
        lines.append("  acc[r][t] += sc * s1 + bi * s2; }")
    elif fmt == "q4":
        lines.append(f"const uint4 q = *(const device uint4*)({name} + (size_t){rowvar} * ({K} / 8) + {kvar} / 8);")
        if gs >= 32:
            lines.append(f"const size_t gi = (size_t){rowvar} * ({K} / {gs}) + {kvar} / {gs};")
            lines.append(f"const float sc = float({name}_s[gi]), bi = float({name}_b[gi]);")
        lines.append("float4 f[8];")
        lines.append(
            "for (int j = 0; j < 4; ++j) { const uint u = q[j]; "
            "f[2*j] = float4(u & 0xfu, (u >> 4) & 0xfu, (u >> 8) & 0xfu, (u >> 12) & 0xfu); "
            "f[2*j+1] = float4((u >> 16) & 0xfu, (u >> 20) & 0xfu, (u >> 24) & 0xfu, u >> 28); }"
        )
        lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0, s2 = 0;")
        lines.append(f"  for (int j = 0; j < 8; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}")
        lines.append("  acc[r][t] += sc * s1 + bi * s2; }")
    return "\n".join(lines)


def _reg_load(fmt, name, K, gs):
    """Load one 16-byte chunk of row lr at column k into wv[c] (and the group scale / bias)."""
    if fmt in ("bf16", "f16", "f32"):
        return f"wv[c] = *(const device uint4*)({name} + (size_t)lr * {K} + k);"
    div = 4 if fmt == "q8" else 8
    return (
        f"wv[c] = *(const device uint4*)({name} + (size_t)lr * ({K} / {div}) + k / {div}); "
        f"sbv[c] = float2({name}_s[(size_t)lr * ({K} / {gs}) + k / {gs}], {name}_b[(size_t)lr * ({K} / {gs}) + k / {gs}]);"
    )


def _reg_fma(fmt, xexpr, T):
    """acc[t] += dot(chunk wv[c] at column k, x[t, k:k+chunk])."""
    if fmt in ("bf16", "f16"):
        vt = "bfloat4" if fmt == "bf16" else "half4"
        return (
            f"{{ const uint4 u = wv[c]; const float4 a0 = float4(as_type<{vt}>(uint2(u.x, u.y))), "
            f"a1 = float4(as_type<{vt}>(uint2(u.z, u.w)));\n"
            f"for (int t = 0; t < {T}; ++t) acc[t] += dot(a0, {xexpr('t', 0)}) + dot(a1, {xexpr('t', 1)}); }}"
        )
    if fmt == "f32":
        return f"{{ const float4 a0 = as_type<float4>(wv[c]); for (int t = 0; t < {T}; ++t) acc[t] += dot(a0, {xexpr('t', 0)}); }}"
    if fmt == "q8":
        return (
            "{ const uint4 u = wv[c]; float4 f[4];\n"
            "for (int j = 0; j < 4; ++j) { const uint q = u[j]; f[j] = float4(q & 0xffu, (q >> 8) & 0xffu, (q >> 16) & 0xffu, q >> 24); }\n"
            f"for (int t = 0; t < {T}; ++t) {{ float s1 = 0, s2 = 0;\n"
            f"  for (int j = 0; j < 4; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}\n"
            "  acc[t] += sbv[c].x * s1 + sbv[c].y * s2; } }"
        )
    return (
        "{ const uint4 u = wv[c]; float4 f[8];\n"
        "for (int j = 0; j < 4; ++j) { const uint q = u[j]; "
        "f[2*j] = float4(q & 0xfu, (q >> 4) & 0xfu, (q >> 8) & 0xfu, (q >> 12) & 0xfu); "
        "f[2*j+1] = float4((q >> 16) & 0xfu, (q >> 20) & 0xfu, (q >> 24) & 0xfu, q >> 28); }\n"
        f"for (int t = 0; t < {T}; ++t) {{ float s1 = 0, s2 = 0;\n"
        f"  for (int j = 0; j < 8; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}\n"
        "  acc[t] += sbv[c].x * s1 + sbv[c].y * s2; } }"
    )


# --------------------------------------------------------------------------
# GEMV with fused prologue
# --------------------------------------------------------------------------

_kernels = {}


def _geo_prologue(K, T, LI):
    """xs[t] = geodesic(xin[t], g[t]); also written to xout by threadgroup 0.
    x and g are read once into registers (EPT elements per thread)."""
    if os.environ.get("OLALA_NOGEO") == "1":  # profiling only: same inputs, no math
        return f"""
    for (uint i = tid; i < {T} * {K}; i += {NSG} * 32) {{ xs[i] = xin[i] + g[i] * geo[0]; if (tgid == 0) xout[i] = xs[i]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """
    NT = NSG * 32
    EPT = (K + NT - 1) // NT
    return f"""
    {{
    threadgroup float red[3][{NSG}];
    const float scale = geo[0], gbias = geo[1];
    for (int t = 0; t < {T}; ++t) {{
        float av[{EPT}], bv[{EPT}];
        float sxx = 0, sxg = 0;
        for (int j = 0; j < {EPT}; ++j) {{
            const uint i = tid + j * {NT};
            av[j] = i < {K} ? xin[t * {K} + i] : 0.0f;
            bv[j] = i < {K} ? g[t * {K} + i] : 0.0f;
            sxx += av[j] * av[j]; sxg += av[j] * bv[j];
        }}
        sxx = simd_sum(sxx); sxg = simd_sum(sxg);
        if (lane == 0) {{ red[0][sg] = sxx; red[1][sg] = sxg; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sxx = 0; sxg = 0;
        for (int i = 0; i < {NSG}; ++i) {{ sxx += red[0][i]; sxg += red[1][i]; }}
        const float xx = max(sxx, 1e-12f);
        const float proj = sxg / xx;
        float sgg = 0;
        for (int j = 0; j < {EPT}; ++j) {{ const float gr = bv[j] - proj * av[j]; sgg += gr * gr; }}
        sgg = simd_sum(sgg);
        if (lane == 0) red[2][sg] = sgg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sgg = 0;
        for (int i = 0; i < {NSG}; ++i) sgg += red[2][i];
        const float tn = max(sqrt(sgg), 1e-8f);
        const float rr = max(sqrt(xx), 1e-6f);
        float theta = min(tn / rr, {math.pi / 4}f);
        theta = min((theta * scale + gbias) / {LI + 1}.0f, {math.pi / 4}f);
        const float c = cos(theta), s = sin(theta) * rr / tn;
        for (int j = 0; j < {EPT}; ++j) {{
            const uint i = tid + j * {NT};
            if (i < {K}) {{
                const float v = av[j] * c + (bv[j] - proj * av[j]) * s;
                xs[t * {K} + i] = v;
                if (tgid == 0) xout[t * {K} + i] = v;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    }}
    """


def _norm_prologue(K, T):
    return f"""
    {{
    threadgroup float red[{NSG}];
    for (int t = 0; t < {T}; ++t) {{
        float sxx = 0;
        for (uint i = tid; i < {K}; i += {NSG} * 32) {{ const float a = xin[t * {K} + i]; sxx += a * a; }}
        sxx = simd_sum(sxx);
        if (lane == 0) red[sg] = sxx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sxx = 0;
        for (int i = 0; i < {NSG}; ++i) sxx += red[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = 1.0f / max(sqrt(sxx), 1e-12f);
        for (uint i = tid; i < {K}; i += {NSG} * 32) {{
            const float v = xin[t * {K} + i] * inv;
            xs[t * {K} + i] = v;
            if (tgid == 0) xout[t * {K} + i] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    }}
    """


def _relu2cat_prologue(K, T, Ka, sa, sb):
    """xs[t, k] = a[t, k] for k < Ka, relu(b[t, k])^2 otherwise; a / b have row strides sa / sb."""
    return f"""
    for (uint i = tid; i < {T} * {K}; i += {NSG} * 32) {{
        const uint t = i / {K}, k = i % {K};
        float v;
        if (k < {Ka}) v = a[t * {sa} + k];
        else {{ const float u = max(b[t * {sb} + k], 0.0f); v = u * u; }}
        xs[i] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """


def gemv_kernel(K, T, segs, prologue, LI=0, Ka=0, sa=0, sb=0, x_in_tg=True, RPS=4):
    """segs: list of (fmt, gs, rows). Output j is (T, rows_j) f32."""
    key = ("gemv", K, T, tuple(segs), prologue, LI, Ka, sa, sb, x_in_tg, RPS)
    if key in _kernels:
        return _kernels[key]
    fmts = [s[0] for s in segs]
    step = max(_chunk(f) for f in fmts)
    for f in fmts:
        if K % _chunk(f):
            raise ValueError("K not a multiple of the load size")
    inputs = []
    if prologue == "geo":
        inputs += ["xin", "g", "geo"]
    elif prologue == "norm":
        inputs += ["xin"]
    elif prologue == "relu2cat":
        inputs += ["a", "b"]
    elif prologue == "moecat":
        inputs += ["parts", "b"]
    else:
        inputs += ["x"]
    for j, (fmt, gs, rows) in enumerate(segs):
        inputs.append(f"w{j}")
        if fmt.startswith("q"):
            inputs += [f"w{j}_s", f"w{j}_b"]
    outputs = [f"y{j}" for j in range(len(segs))]
    if prologue in ("geo", "norm"):
        outputs.append("xout")

    if prologue == "relu2cat":
        decl = ""

        def xexpr(t, j):
            return (
                f"((k + 4 * ({j}) < {Ka}) ? *(const device float4*)(a + {t} * {sa} + k + 4 * ({j})) : "
                f"relu2(*(const device float4*)(b + {t} * {sb} + k + 4 * ({j}))))"
            )
    elif x_in_tg:
        decl = f"threadgroup float xs[{T} * {K}];"

        def xexpr(t, j):
            return f"*(const threadgroup float4*)(xs + {t} * {K} + k + 4 * ({j}))"
    else:
        decl = ""

        def xexpr(t, j):
            return f"*(const device float4*)(x + {t} * {K} + k + 4 * ({j}))"

    pro = {
        "geo": lambda: _geo_prologue(K, T, LI),
        "norm": lambda: _norm_prologue(K, T),
        "relu2cat": lambda: "",
        "moecat": lambda: f"""
    for (uint i = tid; i < {T} * {K}; i += {NSG} * 32) {{
        const uint t = i / {K}, k = i % {K};
        float v;
        if (k < {Ka}) {{ v = 0; for (int p = 0; p < {sa}; ++p) v += parts[(t * {sa} + p) * {Ka} + k]; }}
        else {{ const float u = max(b[t * {sb} + k], 0.0f); v = u * u; }}
        xs[i] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """,
        "plain": lambda: (
            f"for (uint i = tid; i < {T} * {K}; i += {NSG} * 32) xs[i] = x[i];\n"
            "threadgroup_barrier(mem_flags::mem_threadgroup);"
            if x_in_tg
            else ""
        ),
    }[prologue]()

    total = sum(s[2] for s in segs)
    if RPS == 1 and os.environ.get("OLALA_PRELOAD", "0") == "1":
        return _gemv_kernel_preload(key, K, T, segs, inputs, outputs, decl, pro, xexpr, total, step)
    body = []
    off = 0
    for j, (fmt, gs, rows) in enumerate(segs):
        ch = _chunk(fmt)
        body.append(f"""
        if (row >= {off} && row < {off + rows}) {{
            const int lr = row - {off};
            for (int k = lane * {ch}; k < {K}; k += 32 * {ch}) {{
                {_dot_code(fmt, f"w{j}", K, gs, xexpr, T, "lr", "k")}
            }}
        }}""")
        off += rows
    body = "\n".join(body)
    outs = []
    off = 0
    for j, (fmt, gs, rows) in enumerate(segs):
        outs.append(f"if (row >= {off} && row < {off + rows}) y{j}[t * {rows} + row - {off}] = v;")
        off += rows
    outs = "\n".join(outs)

    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint tgid = threadgroup_position_in_grid.x;
    {decl}
    {pro}
    const int row0 = (tgid * {NSG} + sg) * {RPS};
    float acc[{RPS}][{T}];
    for (int r = 0; r < {RPS}; ++r) for (int t = 0; t < {T}; ++t) acc[r][t] = 0;
    for (int r = 0; r < {RPS}; ++r) {{
        const int row = row0 + r;
        {body}
    }}
    for (int r = 0; r < {RPS}; ++r) {{
        const int row = row0 + r;
        if (row >= {total}) break;
        for (int t = 0; t < {T}; ++t) {{
            const float v = simd_sum(acc[r][t]);
            if (lane == 0) {{ {outs} }}
        }}
    }}
    """
    k = mx.fast.metal_kernel(
        name=f"olala_gemv_{abs(hash(key)) % 10**12}",
        input_names=inputs,
        output_names=outputs,
        source=src,
        header=_HDR,
    )
    _kernels[key] = (k, total, step)
    return _kernels[key]


def _gemv_kernel_preload(key, K, T, segs, inputs, outputs, decl, pro, xexpr, total, step):
    """One row per simdgroup; every weight chunk of the row is loaded into registers before
    the prologue runs, so the DRAM latency overlaps the prologue."""
    maxch = max((K + _chunk(f) * 32 - 1) // (_chunk(f) * 32) for f, _, _ in segs)
    pre, body, outs = [], [], []
    off = 0
    for j, (fmt, gs, rows) in enumerate(segs):
        ch = _chunk(fmt)
        nch = (K + ch * 32 - 1) // (ch * 32)
        cond = f"row >= {off} && row < {off + rows}"
        pre.append(f"""
        if ({cond}) {{
            const int lr = row - {off};
            for (int c = 0; c < {nch}; ++c) {{ const int k = (lane + 32 * c) * {ch}; if (k < {K}) {{ {_reg_load(fmt, f"w{j}", K, gs)} }} }}
        }}""")
        body.append(f"""
        if ({cond}) {{
            for (int c = 0; c < {nch}; ++c) {{ const int k = (lane + 32 * c) * {ch}; if (k < {K}) {_reg_fma(fmt, xexpr, T)} }}
        }}""")
        outs.append(f"if ({cond}) y{j}[t * {rows} + row - {off}] = v;")
        off += rows
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint tgid = threadgroup_position_in_grid.x;
    const int row = tgid * {NSG} + sg;
    uint4 wv[{maxch}];
    float2 sbv[{maxch}];
    {"".join(pre)}
    {decl}
    {pro}
    float acc[{T}];
    for (int t = 0; t < {T}; ++t) acc[t] = 0;
    {"".join(body)}
    if (row < {total}) {{
        for (int t = 0; t < {T}; ++t) {{
            const float v = simd_sum(acc[t]);
            if (lane == 0) {{ {" ".join(outs)} }}
        }}
    }}
    """
    k = mx.fast.metal_kernel(
        name=f"olala_gemvp_{abs(hash(key)) % 10**12}",
        input_names=inputs,
        output_names=outputs,
        source=src,
        header=_HDR,
    )
    _kernels[key] = (k, total, step)
    return _kernels[key]


def gemv(x_inputs, ws, prologue="plain", LI=0, Ka=0, sa=0, sb=0, T=1, K=None, RPS=None):
    RPS = RPS or int(os.environ.get("OLALA_RPS", "1"))
    """Run a fused GEMV. x_inputs: list of input arrays for the prologue. ws: list of W.
    Returns [y_j (T, rows_j) f32 ...] (+ xout (T, K) for geo / norm)."""
    K = K or ws[0].cols
    segs = [(w.fmt, w.gs or 0, w.rows) for w in ws]
    if prologue in ("geo", "norm") and T * K * 4 > TG_X_BYTES:
        # too wide for threadgroup memory: run the prologue on its own, then a plain GEMV
        if prologue == "geo":
            xs = geodesic(x_inputs[0], x_inputs[1], x_inputs[2], LI, T, K)
        else:
            x = x_inputs[0]
            xs = x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)
        return gemv([xs], ws, T=T, K=K, RPS=RPS) + [xs]
    if prologue == "moecat" and (T * K * 4 > TG_X_BYTES or os.environ.get("OLALA_MOE_REDUCE") == "1"):
        parts, b = x_inputs
        e = parts.sum(axis=1)  # (T, Ka)
        return gemv([e, b], ws, prologue="relu2cat", Ka=Ka, sa=Ka, sb=sb, T=T, K=K, RPS=RPS)
    x_in_tg = prologue in ("geo", "norm", "moecat") or (
        prologue == "plain" and T * K * 4 <= TG_X_BYTES and os.environ.get("OLALA_PLAIN_TG", "0") == "1")
    kern, total, _ = gemv_kernel(K, T, segs, prologue, LI, Ka, sa, sb, x_in_tg, RPS)
    ntg = (total + NSG * RPS - 1) // (NSG * RPS)
    if _TINY and _tiny():
        ntg = 1
    inputs = list(x_inputs)
    for w in ws:
        inputs += w.arrays()
    shapes = [(T, w.rows) for w in ws]
    dtypes = [mx.float32] * len(ws)
    if prologue in ("geo", "norm"):
        shapes.append((T, K))
        dtypes.append(mx.float32)
    return kern(
        inputs=inputs,
        grid=(ntg * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


# --------------------------------------------------------------------------
# geodesic update as a standalone kernel (for wide outputs like lm_head)
# --------------------------------------------------------------------------


def geodesic(xin, g, geo, LI, T, K):
    key = ("geo", K, T, LI)
    if key not in _kernels:
        src = f"""
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint tgid = threadgroup_position_in_grid.x;
        threadgroup float xs[1];  // unused
        device float* xs_ = xout;
        """ + _geo_prologue(K, T, LI).replace("xs[t * " + str(K) + " + i] = v;", "")
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_geo_{K}_{T}_{LI}", input_names=["xin", "g", "geo"], output_names=["xout"],
            source=src, header=_HDR,
        )
    return _kernels[key](
        inputs=[xin, g, geo], grid=(NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(T, K)], output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# Mamba-3 MIMO decode step
# --------------------------------------------------------------------------

M3_PB = int(os.environ.get("OLALA_M3_PB", "64"))  # state rows (p) per threadgroup


def m3_param_pack(m):
    """f32 [dt_bias (H) | D (H) | b_norm (N) | c_norm (N) | b_bias | c_bias (H*R*N) | mimo_x | mimo_z | mimo_o (H*R*P)]."""
    f32 = mx.float32
    bn = m.B_norm.norm.weight.astype(f32)
    cn = m.C_norm.norm.weight.astype(f32)
    if m.zc:
        bn, cn = 1.0 + bn, 1.0 + cn
    parts = [m.dt_bias, m.D, bn, cn, m.B_bias, m.C_bias, m.in_proj_mimo_x, m.in_proj_mimo_z, m.out_proj_mimo]
    return mx.concatenate([a.astype(f32).reshape(-1) for a in parts])


def m3_step(proj, prm, st, H, P, N, R, A, n_static):
    """proj: (T, n_static + n_dyn) f32 (one token per batch row); prm: m3_param_pack();
    st: packed state (T, H, P*N + R*N + P + A) f32. Returns y (T, H*P) and the new state.

    Threadgroup = (row b, head h, block of PB state rows), 16 simdgroups. Simdgroups 0..2R-1
    normalize / bias / rotate one row of [B | C] each in registers; the others build v, w and
    the previous v for the block; every device load is issued before the single barrier; then
    one simdgroup sweeps one state row (lane = 4 consecutive state columns)."""
    T = proj.shape[0]
    SW = P * N + R * N + P + A
    PB = M3_PB
    NSG_ = 16
    key = ("m3v2", T, H, P, N, R, A, n_static, PB)
    if key not in _kernels:
        assert N == 128 and A == 32 and 2 * R <= NSG_ and PB % NSG_ == 0
        src = f"""
        constexpr int H = {H}, P = {P}, N = {N}, R = {R}, A = {A}, PB = {PB}, NSGM = {NSG_};
        constexpr int PH = 2 * P + 3, NS = {n_static}, PW = {n_static} + 2 * R * N + A;
        constexpr int SW = {SW}, OK_ = P * N, OX = OK_ + R * N, OA = OX + P;
        constexpr int O_DT = 0, O_D = H, O_BN = 2 * H, O_CN = O_BN + N, O_BB = O_CN + N, O_CB = O_BB + H * R * N;
        constexpr int O_MX = O_CB + H * R * N, O_MZ = O_MX + H * R * P, O_MO = O_MZ + H * R * P;
        constexpr int RPSG = PB / NSGM;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int npb = P / PB;
        const int b = blk / (H * npb);
        const int h = (blk / npb) % H;
        const int pb = blk % npb;
        const device float* pr = proj + (size_t)b * PW;
        const device float* ph = pr + h * PH;
        const device float* dyn = pr + NS;
        const device float* sin_ = st + ((size_t)b * H + h) * SW;
        device float* sout = st_out + ((size_t)b * H + h) * SW;

        threadgroup float4 kqs[2 * R][N / 4];
        threadgroup float vws[3][R][PB];

        // early loads for the sweep (independent of the prep)
        float4 srow[RPSG];
        for (int i = 0; i < RPSG; ++i) srow[i] = ((const device float4*)(sin_ + (pb * PB + sg + NSGM * i) * N))[lane];
        float4 kp[R];
        for (int r = 0; r < R; ++r) kp[r] = ((const device float4*)(sin_ + OK_ + r * N))[lane];

        const float dt = olala_softplus(ph[2 * P] + prm[O_DT + h]);
        const float a = min(-olala_softplus(ph[2 * P + 1]), -1e-4f);
        const float alpha = exp(a * dt);
        const float trap = olala_sigmoid(ph[2 * P + 2]);
        const float beta = (1.0f - trap) * dt * alpha;
        const float gamma = trap * dt;

        if (sg < 2 * R) {{
            const int gi = sg;
            const bool isq = gi >= R;
            const int r = isq ? gi - R : gi;
            const float4 u = ((const device float4*)(dyn + gi * N))[lane];
            const float4 wn = ((const device float4*)(prm + (isq ? O_CN : O_BN)))[lane];
            const float4 bb = ((const device float4*)(prm + (isq ? O_CB : O_BB) + (h * R + r) * N))[lane];
            // rotary angle of this lane (A == 32): wrapped cumulative angle
            float thl = sin_[OA + lane] + tanh(dyn[2 * R * N + lane]) * {math.pi}f * dt;
            thl = thl - {2 * math.pi}f * rint(thl * {0.5 / math.pi}f);
            if (gi == 0 && pb == 0) sout[OA + lane] = thl;
            const float ss = simd_sum(dot(u, u));
            float4 val = u * rsqrt(ss / N + {1e-6}f) * wn + bb;
            // rotate pairs (n, n + 64) for n < 32: lanes 0..7 hold n = 0..31, lanes 16..23 hold n = 64..95
            const float4 partner = simd_shuffle_xor(val, 16);
            const int base = 4 * (lane & 15);
            const float4 th4 = float4(simd_shuffle(thl, (base + 0) & 31), simd_shuffle(thl, (base + 1) & 31),
                                      simd_shuffle(thl, (base + 2) & 31), simd_shuffle(thl, (base + 3) & 31));
            if ((lane & 15) < 8) {{
                const float4 c = cos(th4), s = sin(th4);
                val = (lane < 16) ? (val * c - partner * s) : (partner * s + val * c);
            }}
            kqs[gi][lane] = val;
            if (!isq && pb == 0) ((device float4*)(sout + OK_ + r * N))[lane] = val;
        }} else {{
            for (int i = tid - 2 * R * 32; i < R * PB; i += (NSGM - 2 * R) * 32) {{
                const int r = i / PB, pl = i % PB, p = pb * PB + pl;
                const float mxv = prm[O_MX + (h * R + r) * P + p];
                const float xv = ph[P + p];
                const float zz = ph[p] * prm[O_MZ + (h * R + r) * P + p];
                vws[0][r][pl] = xv * mxv;
                vws[1][r][pl] = zz / (1.0f + exp(-zz)) * prm[O_MO + (h * R + r) * P + p];
                vws[2][r][pl] = sin_[OX + p] * mxv;
                if (r == 0) sout[OX + p] = xv;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 kr[R], qr[R];
        for (int r = 0; r < R; ++r) {{ kr[r] = kqs[r][lane]; qr[r] = kqs[R + r][lane]; }}
        const float Dh = prm[O_D + h];
        for (int i = 0; i < RPSG; ++i) {{
            const int pl = sg + NSGM * i, p = pb * PB + pl;
            float4 kv = 0, kvp = 0, qw = 0;
            float d = 0;
            for (int r = 0; r < R; ++r) {{
                const float vv = vws[0][r][pl], ww = vws[1][r][pl], vp = vws[2][r][pl];
                kv += vv * kr[r]; kvp += vp * kp[r]; qw += ww * qr[r];
                d += ww * vv;
            }}
            const float4 s = alpha * srow[i] + beta * kvp + gamma * kv;
            ((device float4*)(sout + p * N))[lane] = s;
            const float acc = simd_sum(dot(s, qw));
            if (lane == 0) y[(size_t)b * H * P + h * P + p] = acc + Dh * d;
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_m3v2_{T}_{PB}",
            input_names=["proj", "prm", "st"],
            output_names=["y", "st_out"],
            source=src,
            header=_HDR,
        )
    kern = _kernels[key]
    npb = P // PB
    return kern(
        inputs=[proj, prm, st],
        grid=(NSG_ * 32 if _tiny("m3") else T * H * npb * NSG_ * 32, 1, 1),
        threadgroup=(NSG_ * 32, 1, 1),
        output_shapes=[(T, H * P), (T, H, SW)],
        output_dtypes=[mx.float32] * 2,
    )


# --------------------------------------------------------------------------
# MoE experts
# --------------------------------------------------------------------------


def _expert_dot(fmt, gs, Kin, xexpr, rowexpr):
    """acc += W_e[row, k:k+chunk] . x[k:k+chunk] for an expert tensor (E, rows, Kin)."""
    return _dot_code(fmt, "w", Kin, gs, xexpr, 1, rowexpr, "k").replace("acc[r][t]", "acc")


def moe_up(logits, ebias, x0, x0_stride, we: W, top_k, scaling, T):
    """h[t, k, :] = relu(W1[e_k] @ x0[t])^2 * p_k ; returns h (T, K, F) f32, inds (T, K) int32."""
    E, F, D = we.w.shape[0], we.rows, we.cols
    ch = _chunk(we.fmt)
    LPR = 8 if (D // ch) % 8 == 0 else 4  # lanes per row
    RPSG = 32 // LPR  # rows per simdgroup
    RTG = RPSG * NSG  # rows per threadgroup
    key = ("moe_up", E, F, D, we.key(), top_k, T)
    if key not in _kernels:
        def xexpr(t, j):
            return f"*(const threadgroup float4*)(xs + k + 4 * ({j}))"
        dot = _dot_code(we.fmt, "w", D, we.gs or 0, xexpr, 1, "erow", "k").replace("acc[r][t]", "acc")
        src = f"""
        constexpr int E = {E}, F = {F}, D = {D}, TOPK = {top_k}, LPR = {LPR}, RPSG = {RPSG}, RTG = {RTG};
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int nblk = (F + RTG - 1) / RTG;
        const int t = blk / (TOPK * nblk);
        const int kk = (blk / nblk) % TOPK;
        const int fb = blk % nblk;
        threadgroup float xs[D];
        threadgroup int sel_idx[TOPK];
        threadgroup float sel_p[TOPK];
        for (int i = tid; i < D; i += {NSG} * 32) xs[i] = x0[t * {x0_stride} + i];
        if (sg == 0) {{
            // top-k over E experts: lane holds E / 32 candidates
            float sc[E / 32], sv[E / 32];
            for (int j = 0; j < E / 32; ++j) {{
                const int e = lane + 32 * j;
                sv[j] = olala_sigmoid(logits[t * E + e]);
                sc[j] = sv[j] + ebias[e];
            }}
            float psum = 0;
            for (int q = 0; q < TOPK; ++q) {{
                float best = -INFINITY; int bj = 0;
                for (int j = 0; j < E / 32; ++j) if (sc[j] > best) {{ best = sc[j]; bj = j; }}
                const float m = simd_max(best);
                // smallest expert index among the lanes holding the max
                const int cand = (best == m) ? (lane + 32 * bj) : 0x7fffffff;
                const int e = simd_min(cand);
                float pv = 0;
                if (e == lane + 32 * bj) {{ pv = sv[bj]; sc[bj] = -INFINITY; }}
                pv = simd_sum(pv);
                if (lane == 0) {{ sel_idx[q] = e; sel_p[q] = pv; }}
                psum += pv;
            }}
            if (lane == 0) for (int q = 0; q < TOPK; ++q) sel_p[q] = sel_p[q] / (psum + 1e-20f) * {scaling}f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int e = sel_idx[kk];
        const float pk = sel_p[kk];
        if (blk % nblk == 0 && tid == 0) inds[t * TOPK + kk] = e;
        const int sub = lane / LPR, sl = lane % LPR;
        const int row = fb * RTG + sg * RPSG + sub;
        float acc = 0;
        if (row < F) {{
            const int erow = e * F + row;
            for (int k = sl * {ch}; k < D; k += LPR * {ch}) {{
                {dot}
            }}
        }}
        for (int o = LPR / 2; o > 0; o /= 2) acc += simd_shuffle_xor(acc, o);
        if (sl == 0 && row < F) {{
            const float u = max(acc, 0.0f);
            hout[((size_t)t * TOPK + kk) * F + row] = u * u * pk;
        }}
        """
        names = ["logits", "ebias", "x0", "w"] + (["w_s", "w_b"] if we.fmt.startswith("q") else [])
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_moe_up_{abs(hash(key)) % 10**12}", input_names=names,
            output_names=["hout", "inds"], source=src, header=_HDR,
        )
    nblk = (F + RTG - 1) // RTG
    return _kernels[key](
        inputs=[logits, ebias, x0] + we.arrays(),
        grid=(NSG * 32 if _tiny("moe_up") else T * top_k * nblk * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(T, top_k, F), (T, top_k)],
        output_dtypes=[mx.float32, mx.int32],
    )


def _topk_code(top_k, scaling):
    if os.environ.get("OLALA_FAKE_TOPK") == "1":  # profiling only
        return f"if (lane < {top_k}) {{ sel_idx[lane] = lane * 37 + int(logits[t * E] > 1e30f); sel_p[lane] = 0.4f; }}"
    return f"""
            float sc[E / 32], sv[E / 32];
            for (int j = 0; j < E / 32; ++j) {{
                const int e = lane + 32 * j;
                sv[j] = olala_sigmoid(logits[t * E + e]);
                sc[j] = sv[j] + ebias[e];
            }}
            float psum = 0;
            for (int q = 0; q < TOPK; ++q) {{
                float best = -INFINITY; int bj = 0;
                for (int j = 0; j < E / 32; ++j) if (sc[j] > best) {{ best = sc[j]; bj = j; }}
                const float m = simd_max(best);
                const int cand = (best == m) ? (lane + 32 * bj) : 0x7fffffff;
                const int e = simd_min(cand);
                float pv = 0;
                if (e == lane + 32 * bj) {{ pv = sv[bj]; sc[bj] = -INFINITY; }}
                pv = simd_sum(pv);
                if (lane == 0) {{ sel_idx[q] = e; sel_p[q] = pv; }}
                psum += pv;
            }}
            if (lane == 0) for (int q = 0; q < TOPK; ++q) sel_p[q] = sel_p[q] / (psum + 1e-20f) * {scaling}f;"""


# Routed-expert outputs: 1 = float atomics into one (T, D) buffer (fastest; the summation order
# of the 72 partials per output varies between runs), 0 = per-threadgroup partials summed in a
# fixed order by the next GEMV (bit-reproducible). OLALA_DETERMINISTIC=1 selects 0.
_MOE_ATOMIC = int(os.environ.get("OLALA_MOE_ATOMIC", "0" if os.environ.get("OLALA_DETERMINISTIC") == "1" else "1"))


def _last_block_code():
    """The last threadgroup of token t to finish sums the TOPK * NB partials in a fixed order."""
    return """
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup bool is_last;
        if (tid == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            const uint done = atomic_fetch_add_explicit(&ctr[t], 1u, memory_order_relaxed);
            is_last = done == TOPK * NB - 1;
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_last) {
            for (int o = tid; o < D; o += NT) {
                float v = 0;
                for (int p = 0; p < TOPK * NB; ++p) v += atomic_load_explicit(&parts[((size_t)t * TOPK * NB + p) * D + o], memory_order_relaxed);
                atomic_store_explicit(&eout[(size_t)t * D + o], v, memory_order_relaxed);
            }
        }
    """
MOE_FB = int(os.environ.get("OLALA_MOE_FB", "64"))  # expert hidden units per threadgroup in moe_fused


def moe_fused(logits, ebias, x0, x0_stride, wu: W, wd: W, top_k, scaling, T):
    """Routed experts in one kernel. Threadgroup (t, k, block) computes
    h = relu(W1[e_k][block] x0[t])^2 * p_k for MOE_FB hidden units and its contribution
    W2[e_k][:, block] h to the expert output. Returns partial sums (T, top_k * NB, D).

    Latency-oriented: simdgroup 0 selects the experts while the others stage x0; phase 1
    uses LPRU lanes per hidden unit in a single pass; phase 2 gives each thread whole
    outputs, so every weight load of a phase is independent."""
    E, F, D = wu.w.shape[0], wu.rows, wu.cols
    FB = MOE_FB
    NB = F // FB
    chu, chd = _chunk(wu.fmt), _chunk(wd.fmt)
    NT = NSG * 32
    LPRU = max(1, min(32, (NT * (D // chu)) // (FB * max(1, D // chu)) if False else NT // FB))
    while LPRU > 1 and (D // chu) % LPRU:
        LPRU //= 2
    key = ("moe_fused2", E, F, D, wu.key(), wd.key(), top_k, T, FB, NSG)
    if key not in _kernels:
        def xu(t, j):
            return f"*(const threadgroup float4*)(xs + k + 4 * ({j}))"

        def xd(t, j):
            return f"*(const threadgroup float4*)(hs + (k - f0) + 4 * ({j}))"
        dotu = _dot_code(wu.fmt, "wu", D, wu.gs or 0, xu, 1, "erow", "k").replace("acc[r][t]", "acc")
        dotd = _dot_code(wd.fmt, "wd", F, wd.gs or 0, xd, 1, "erow", "k").replace("acc[r][t]", "acc")
        src = f"""
        constexpr int E = {E}, F = {F}, D = {D}, TOPK = {top_k}, FB = {FB}, NB = {NB}, NT = {NT}, LPRU = {LPRU};
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / (TOPK * NB);
        const int kk = (blk / NB) % TOPK;
        const int fb = blk % NB;
        const int f0 = fb * FB;
        threadgroup float xs[D];
        threadgroup float hs[FB];
        threadgroup int sel_idx[TOPK];
        threadgroup float sel_p[TOPK];
        if (sg == 0) {{
            {_topk_code(top_k, scaling)}
        }} else {{
            for (int i = tid - 32; i < D; i += NT - 32) xs[i] = x0[t * {x0_stride} + i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int e = sel_idx[kk];
        const float pk = sel_p[kk];
        // phase 1: hidden unit f = tid / LPRU, LPRU lanes split D
        {{
            const int f = tid / LPRU, sl = tid % LPRU;
            float acc = 0;
            if (f < FB) {{
                const int erow = e * F + f0 + f;
                for (int k = sl * {chu}; k < D; k += LPRU * {chu}) {{
                    {dotu}
                }}
            }}
            for (int o = LPRU / 2; o > 0; o /= 2) acc += simd_shuffle_xor(acc, o);
            if (sl == 0 && f < FB) {{ const float u = max(acc, 0.0f); hs[f] = u * u * pk; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // phase 2: thread o computes output o over this block's FB hidden units
        for (int o = tid; o < D; o += NT) {{
            const int erow = e * D + o;
            float acc = 0;
            for (int k = f0; k < f0 + FB; k += {chd}) {{
                {dotd}
            }}
            {"atomic_fetch_add_explicit(&parts[(size_t)t * D + o], acc, memory_order_relaxed);" if _MOE_ATOMIC == 1 else
             "atomic_store_explicit(&parts[((size_t)t * TOPK * NB + kk * NB + fb) * D + o], acc, memory_order_relaxed);" if _MOE_ATOMIC == 2 else
             "parts[((size_t)t * TOPK * NB + kk * NB + fb) * D + o] = acc;"}
        }}
        {_last_block_code() if _MOE_ATOMIC == 2 else ""}
        """
        names = ["logits", "ebias", "x0", "wu"] + (["wu_s", "wu_b"] if wu.fmt.startswith("q") else []) + ["wd"] + (
            ["wd_s", "wd_b"] if wd.fmt.startswith("q") else [])
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_moe_fused2_{abs(hash(key)) % 10**12}a{_MOE_ATOMIC}", input_names=names,
            output_names=["parts", "eout", "ctr"] if _MOE_ATOMIC == 2 else ["parts"], source=src, header=_HDR,
            atomic_outputs=bool(_MOE_ATOMIC),
        )
    if _MOE_ATOMIC == 2:
        return _kernels[key](
            inputs=[logits, ebias, x0] + wu.arrays() + wd.arrays(),
            grid=(T * top_k * NB * NSG * 32, 1, 1),
            threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(T, top_k * NB, D), (T, D), (T,)],
            output_dtypes=[mx.float32, mx.float32, mx.uint32],
            init_value=0,
        )[1]
    if _MOE_ATOMIC:
        return _kernels[key](
            inputs=[logits, ebias, x0] + wu.arrays() + wd.arrays(),
            grid=(T * top_k * NB * NSG * 32, 1, 1),
            threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(T, D)],
            output_dtypes=[mx.float32],
            init_value=0,
        )[0]
    return _kernels[key](
        inputs=[logits, ebias, x0] + wu.arrays() + wd.arrays(),
        grid=(NSG * 32 if _tiny("moe") else T * top_k * NB * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(T, top_k * NB, D)],
        output_dtypes=[mx.float32],
    )[0]


def moe_down(h, inds, we: W, top_k, T, RPS=2):
    """e[t, o] = sum_k W2[e_k][o, :] . h[t, k, :]  -> (T, D) f32. One simdgroup per k."""
    E, D, F = we.w.shape[0], we.rows, we.cols
    ch = _chunk(we.fmt)
    key = ("moe_down", E, D, F, we.key(), top_k, T, RPS)
    if key not in _kernels:
        def xexpr(t, j):
            return f"*(const device float4*)(hk + k + 4 * ({j}))"
        dot = _dot_code(we.fmt, "w", F, we.gs or 0, xexpr, 1, "erow", "k").replace("acc[r][t]", "acc[r]")
        src = f"""
        constexpr int D = {D}, F = {F}, TOPK = {top_k}, RPS = {RPS};
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;    // sg = expert slot k
        const uint blk = threadgroup_position_in_grid.x;
        const int nblk = D / RPS;
        const int t = blk / nblk;
        const int o0 = (blk % nblk) * RPS;
        threadgroup float part[TOPK][RPS];
        const int e = inds[t * TOPK + sg];
        const device float* hk = hin + ((size_t)t * TOPK + sg) * F;
        float acc[RPS];
        for (int r = 0; r < RPS; ++r) acc[r] = 0;
        for (int r = 0; r < RPS; ++r) {{
            const int erow = e * D + o0 + r;
            for (int k = lane * {ch}; k < F; k += 32 * {ch}) {{
                {dot}
            }}
        }}
        for (int r = 0; r < RPS; ++r) {{
            const float v = simd_sum(acc[r]);
            if (lane == 0) part[sg][r] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < RPS) {{
            float s = 0;
            for (int k = 0; k < TOPK; ++k) s += part[k][tid];
            out[t * D + o0 + tid] = s;
        }}
        """
        names = ["hin", "inds", "w"] + (["w_s", "w_b"] if we.fmt.startswith("q") else [])
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_moe_down_{abs(hash(key)) % 10**12}", input_names=names,
            output_names=["out"], source=src, header=_HDR,
        )
    return _kernels[key](
        inputs=[h, inds] + we.arrays(),
        grid=(top_k * 32 if _tiny("moe_down") else T * (D // RPS) * top_k * 32, 1, 1),
        threadgroup=(top_k * 32, 1, 1),
        output_shapes=[(T, D)],
        output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# attention (V layers)
# --------------------------------------------------------------------------


def v_prep(big, small, kprev, vprev, prm, pos, Hq, Hk, Dh, Rk, n_noise, window, T, kv_dtype):
    """Returns q (T,Hq,Dh) f32 (normed, scalable-softmax scaled), k / v (T,Hk,1,Dh) kv_dtype,
    raw k / v (T,1,Hk,Dh) f32 for the next token shift."""
    key = ("v_prep", T, Hq, Hk, Dh, Rk, str(kv_dtype))
    if key not in _kernels:
        ot = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[kv_dtype]
        nbig = Hq * Dh + 2 * Rk * Dh + (Hq - Hk) * Dh
        nsmall = 2 * Hk * Rk + 2 * Hk + n_noise
        src = f"""
        constexpr int Hq = {Hq}, Hk = {Hk}, Dh = {Dh}, Rk = {Rk};
        constexpr int NB = {nbig}, NSM = {nsmall};
        const uint tid = thread_position_in_threadgroup.x;   // Dh threads
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / (Hq + Hk);
        const int hh = blk % (Hq + Hk);
        const device float* bg = big + (size_t)t * NB;
        const device float* sm = small + (size_t)t * NSM;
        const float p = float(pos[t]);
        threadgroup float red[Dh / 32];
        float val;
        if (hh < Hq) {{
            val = bg[hh * Dh + tid];
        }} else {{
            const int h = hh - Hq;
            const device float* bk = bg + Hq * Dh;            // W_B_k (Rk, Dh)
            const device float* bv = bk + Rk * Dh;            // W_B_v
            float kr = 0, vr = 0;
            for (int r = 0; r < Rk; ++r) {{
                kr += sm[h * Rk + r] * bk[r * Dh + tid];
                vr += sm[Hk * Rk + h * Rk + r] * bv[r * Dh + tid];
            }}
            kr /= Rk; vr /= Rk;
            float ak = olala_sigmoid(sm[2 * Hk * Rk + h]);
            float av = olala_sigmoid(sm[2 * Hk * Rk + Hk + h]);
            if (p <= 0.0f) {{ ak = 0; av = 0; }}
            const float kp = kprev[((size_t)t * Hk + h) * Dh + tid];
            const float vp = vprev[((size_t)t * Hk + h) * Dh + tid];
            kraw_out[((size_t)t * Hk + h) * Dh + tid] = kr;
            vraw_out[((size_t)t * Hk + h) * Dh + tid] = vr;
            val = ak * kp + (1 - ak) * kr;
            v_out[((size_t)t * Hk + h) * Dh + tid] = ({ot})(av * vp + (1 - av) * vr);
        }}
        float s = simd_sum(val * val);
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = 0;
        for (int i = 0; i < Dh / 32; ++i) s += red[i];
        const float rn = rsqrt(s / Dh + {1e-6}f);
        if (hh < Hq) {{
            const float lp = log(min(p + 1.0f, {float(window)}f));
            q_out[((size_t)t * Hq + hh) * Dh + tid] = val * rn * qn[tid] * scaler[hh] * lp;
        }} else {{
            const int h = hh - Hq;
            k_out[((size_t)t * Hk + h) * Dh + tid] = ({ot})(val * rn * kn[tid]);
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_v_prep_{abs(hash(key)) % 10**12}",
            input_names=["big", "small", "kprev", "vprev", "qn", "kn", "scaler", "pos"],
            output_names=["q_out", "k_out", "v_out", "kraw_out", "vraw_out"],
            source=src, header=_HDR,
        )
    return _kernels[key](
        inputs=[big, small, kprev, vprev, prm["q_norm"], prm["k_norm"], prm["scaler"], pos],
        grid=(Dh if _tiny("v_prep") else T * (Hq + Hk) * Dh, 1, 1),
        threadgroup=(Dh, 1, 1),
        output_shapes=[(T, Hq, Dh), (T, Hk, 1, Dh), (T, Hk, 1, Dh), (T, 1, Hk, Dh), (T, 1, Hk, Dh)],
        output_dtypes=[mx.float32, kv_dtype, kv_dtype, mx.float32, mx.float32],
    )


ATT_BLK = 256  # keys per threadgroup split


def attn_decode(q, keys, values, kstart, scale, softcap, T, Hq, Hk, Dh):
    """q (T,Hq,Dh) f32; keys / values (B,Hk,S,Dh) (possibly strided views).
    Valid keys of row t: [kstart[t], S). Returns partial o (T,Hq,NS,Dh), m, l (T,Hq,NS)."""
    S = keys.shape[2]
    NS = (S + ATT_BLK - 1) // ATT_BLK
    G = Hq // Hk
    key = ("attn", Hq, Hk, Dh, str(keys.dtype))
    if key not in _kernels:
        src = f"""
        constexpr int Hq = {Hq}, Hk = {Hk}, Dh = {Dh}, G = {G}, BLK = {ATT_BLK}, NW = 4;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int S = keys_shape[2];
        const int NS = (S + BLK - 1) / BLK;
        const int t = blk / (Hk * NS);
        const int hk = (blk / NS) % Hk;
        const int sp = blk % NS;
        const int j0 = max(sp * BLK, kstart[t]);
        const int j1 = min((sp + 1) * BLK, S);
        threadgroup float qs[G][Dh];
        threadgroup float mo[NW][G];
        threadgroup float lo[NW][G];
        threadgroup float oo[NW][G][Dh];
        for (int i = tid; i < G * Dh; i += NW * 32) qs[i / Dh][i % Dh] = q[((size_t)t * Hq + hk * G + i / Dh) * Dh + i % Dh];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const size_t kb = (size_t)t * keys_strides[0] + (size_t)hk * keys_strides[1];
        const size_t vb = (size_t)t * values_strides[0] + (size_t)hk * values_strides[1];
        float m[G], l[G], o[G][Dh / 32];
        for (int g = 0; g < G; ++g) {{ m[g] = -INFINITY; l[g] = 0; for (int d = 0; d < Dh / 32; ++d) o[g][d] = 0; }}
        // simdgroup sg handles keys j0 + sg*32 + lane, stepping by NW*32
        for (int jb = j0 + sg * 32; jb < j1; jb += NW * 32) {{
            const int j = jb + lane;
            float sc[G];
            if (j < j1) {{
                const device auto* kr = keys + kb + (size_t)j * keys_strides[2];
                for (int g = 0; g < G; ++g) sc[g] = 0;
                for (int d = 0; d < Dh; d += 4) {{
                    const float4 kv = float4(kr[d], kr[d + 1], kr[d + 2], kr[d + 3]);
                    for (int g = 0; g < G; ++g) sc[g] += dot(kv, *(threadgroup float4*)&qs[g][d]);
                }}
                for (int g = 0; g < G; ++g) sc[g] = tanh(sc[g] * {scale}f / {softcap}f) * {softcap}f;
            }} else {{
                for (int g = 0; g < G; ++g) sc[g] = -INFINITY;
            }}
            float pj[G];
            for (int g = 0; g < G; ++g) {{
                const float bm = simd_max(sc[g]);
                const float mn = max(m[g], bm);
                const float corr = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - mn);
                pj[g] = (sc[g] == -INFINITY) ? 0.0f : exp(sc[g] - mn);
                l[g] = l[g] * corr + simd_sum(pj[g]);
                for (int d = 0; d < Dh / 32; ++d) o[g][d] *= corr;
                m[g] = mn;
            }}
            const int nk = min(32, j1 - jb);
            for (int u = 0; u < nk; ++u) {{
                const device auto* vr = values + vb + (size_t)(jb + u) * values_strides[2];
                float vv[Dh / 32];
                for (int d = 0; d < Dh / 32; ++d) vv[d] = float(vr[lane + 32 * d]);
                for (int g = 0; g < G; ++g) {{
                    const float pu = simd_shuffle(pj[g], u);
                    for (int d = 0; d < Dh / 32; ++d) o[g][d] += pu * vv[d];
                }}
            }}
        }}
        for (int g = 0; g < G; ++g) {{
            if (lane == 0) {{ mo[sg][g] = m[g]; lo[sg][g] = l[g]; }}
            for (int d = 0; d < Dh / 32; ++d) oo[sg][g][lane + 32 * d] = o[g][d];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // combine the NW simdgroups
        for (int i = tid; i < G * Dh; i += NW * 32) {{
            const int g = i / Dh, d = i % Dh;
            float M = -INFINITY;
            for (int w = 0; w < NW; ++w) M = max(M, mo[w][g]);
            float L = 0, O = 0;
            for (int w = 0; w < NW; ++w) {{
                const float c = (mo[w][g] == -INFINITY) ? 0.0f : exp(mo[w][g] - M);
                L += lo[w][g] * c; O += oo[w][g][d] * c;
            }}
            const size_t oi = ((size_t)t * Hq + hk * G + g) * NS + sp;
            o_out[oi * Dh + d] = O;
            if (d == 0) {{ m_out[oi] = M; l_out[oi] = L; }}
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_attn_{abs(hash(key)) % 10**12}",
            input_names=["q", "keys", "values", "kstart"],
            output_names=["o_out", "m_out", "l_out"],
            source=src, header=_HDR, ensure_row_contiguous=False,
        )
    return _kernels[key](
        inputs=[q, keys, values, kstart],
        grid=(128 if _tiny("attn") else T * Hk * NS * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, Hq, NS, Dh), (T, Hq, NS), (T, Hq, NS)],
        output_dtypes=[mx.float32] * 3,
    )


def attn_combine(o, m, l, small, big, Hq, Hk, Dh, snr, n_noise, gate_bias, T, small_off, big_off):
    """Merge the key splits, apply the differential combination and the output gate.
    Returns (T, n_sig * Dh) f32."""
    NS = o.shape[2]
    n_sig = Hq - n_noise
    key = ("attn_comb", Hq, Dh, snr, n_noise, NS, small_off, big_off, small.shape[1], big.shape[1])
    if key not in _kernels:
        src = f"""
        constexpr int Hq = {Hq}, Dh = {Dh}, SNR = {snr}, NN = {n_noise}, NS = {NS}, NSIG = {n_sig};
        const uint d = thread_position_in_threadgroup.x;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / NSIG;
        const int hs = blk % NSIG;            // signal head index
        const int g = hs / SNR, i = hs % SNR;
        const int hsig = g * (SNR + 1) + i, hnoi = g * (SNR + 1) + SNR;
        float res[2];
        for (int u = 0; u < 2; ++u) {{
            const int hq = u == 0 ? hsig : hnoi;
            const size_t base = ((size_t)t * Hq + hq) * NS;
            float M = -INFINITY;
            for (int s = 0; s < NS; ++s) M = max(M, m[base + s]);
            float L = 0, O = 0;
            for (int s = 0; s < NS; ++s) {{
                const float c = (m[base + s] == -INFINITY) ? 0.0f : exp(m[base + s] - M);
                L += l[base + s] * c; O += o[(base + s) * Dh + d] * c;
            }}
            res[u] = O / L;
        }}
        const float lam = olala_sigmoid(small[t * {small.shape[1]} + {small_off} + g]);
        const float gt = big[t * {big.shape[1]} + {big_off} + hs * Dh + d] + {gate_bias}f;
        out[((size_t)t * NSIG + hs) * Dh + d] = (res[0] - lam * res[1]) * (gt / (1.0f + exp(-gt)));
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_attn_comb_{abs(hash(key)) % 10**12}",
            input_names=["o", "m", "l", "small", "big"], output_names=["out"], source=src, header=_HDR,
        )
    return _kernels[key](
        inputs=[o, m, l, small, big],
        grid=(Dh if _tiny("combine") else T * n_sig * Dh, 1, 1),
        threadgroup=(Dh, 1, 1),
        output_shapes=[(T, n_sig * Dh)],
        output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# decode driver
# --------------------------------------------------------------------------


class _Plan:
    """Per-model prepared weights / parameters for the fused decode path."""

    def __init__(self, model):
        a = model.args
        self.layers = []
        f32 = mx.float32
        for layer in model.layers:
            L = {"type": layer.layer_type, "li": layer.layer_idx}
            mx_ = layer.mixer
            if layer.layer_type == "M":
                L["in"] = W(mx_.in_proj)
                L["prm"] = {"m3": m3_param_pack(mx_)}
                L["dims"] = (mx_.H, mx_.P, mx_.N, mx_.R, mx_.A, mx_.n_static)
            else:
                L["in"] = W(mx_.in_proj)
                L["in_small"] = W(mx_.in_proj_small)
                zc = a.zero_centered_gamma
                L["prm"] = {
                    "q_norm": (1.0 + mx_.q_norm.norm.weight.astype(f32)) if zc else mx_.q_norm.norm.weight.astype(f32),
                    "k_norm": (1.0 + mx_.k_norm.norm.weight.astype(f32)) if zc else mx_.k_norm.norm.weight.astype(f32),
                    "scaler": mx_.softmax_scaler.astype(f32),
                }
                L["attn"] = mx_
            L["mixer_proj"] = W(layer.mixer_proj)
            L["geo_mixer"] = mx.concatenate([layer.geodesic_mixer.scale, layer.geodesic_mixer.bias]).astype(f32)
            L["geo_mlp"] = mx.concatenate([layer.geodesic_mlp.scale, layer.geodesic_mlp.bias]).astype(f32)
            moe = layer.mlp
            L["router"] = W(moe.moe_gate)
            L["in_cat"] = W(moe.in_cat)
            L["out_cat"] = W(moe.out_cat)
            L["up"] = W(moe.experts.experts)
            L["down"] = W(moe.experts.output_experts)
            L["ebias"] = moe.expert_bias.astype(f32)
            L["moe"] = moe
            self.layers.append(L)
        mx.eval([[v for v in L["prm"].values()] for L in self.layers])


_plans = {}


def _plan(model):
    p = _plans.get(id(model))
    if p is None or p[0] is not model:
        p = (model, _Plan(model))
        _plans[id(model)] = p
    return p[1]


def can_decode(model, x, cache, ssm_mask) -> bool:
    if not enabled() or os.environ.get("OLALA_FUSED", "1") == "0":
        return False
    B, L, D = x.shape
    if L != 1 or B > MAX_ROWS or ssm_mask is not None or cache is None or cache[0] is None:
        return False
    for i in model.attn_idx:
        c = cache[i]
        if type(c).__name__ not in ("KVCache", "BatchKVCache") or c.keys is None:
            return False
    for i in model.ssm_idx:
        c = cache[i]
        if c[0] is None or getattr(c, "lengths", None) is not None:
            return False
    try:
        _plan(model)
    except (ValueError, KeyError, AttributeError):
        return False
    return True


# profiling: OLALA_TINY=name,... launches those kernels with a single threadgroup (wrong results,
# same launches and dependencies), to measure the cost of their actual work in place
_TINY = set(filter(None, os.environ.get("OLALA_TINY", "").split(",")))
_cur = [""]


def _tiny(name=None):
    return (name or _cur[0]) in _TINY or "all" in _TINY
_SKIP = set(filter(None, os.environ.get("OLALA_SKIP", "").split(",")))
_skip_cache = {}


_STUB = set(filter(None, os.environ.get("OLALA_STUB", "").split(",")))


def _k(name, li, fn, *args, **kw):
    """Profiling hooks: kernels listed in OLALA_SKIP reuse their first outputs; kernels in
    OLALA_STUB are replaced by a trivial op that keeps their dependencies."""
    _cur[0] = name
    if name in _STUB:
        key = (name, li)
        if key not in _skip_cache:
            _skip_cache[key] = fn(*args, **kw)
            mx.eval(_skip_cache[key])
        dep = [a for a in args if isinstance(a, mx.array)] + [a for a in args[0] if isinstance(args[0], list) and isinstance(a, mx.array)] if args else []
        outs = _skip_cache[key]
        dep_arr = next(a for a in (args[0] if isinstance(args[0], list) else args) if isinstance(a, mx.array))
        return [o + 0 * dep_arr.reshape(-1)[:1].astype(o.dtype) if o.ndim else o for o in outs] if isinstance(outs, (list, tuple)) else outs + 0 * dep_arr.reshape(-1)[:1].astype(outs.dtype)
    if name in _SKIP:
        key = (name, li)
        if key not in _skip_cache:
            _skip_cache[key] = fn(*args, **kw)
            mx.eval(_skip_cache[key])
        return _skip_cache[key]
    return fn(*args, **kw)


def _moe_block(L, resid, gmix, T, D):
    """geodesic(resid, gmix) -> MoE. Returns (resid', g) with g the MoE output."""
    li = L["li"]
    moe = L["moe"]
    logits, hcat, resid = _k("in_cat", li, gemv, [resid, gmix, L["geo_mixer"]], [L["router"], L["in_cat"]],
                             prologue="geo", LI=li, T=T, K=D)
    if _MOE_SPLIT:
        hexp, inds = _k("moe_up", li, moe_up, logits, L["ebias"], hcat, hcat.shape[1], L["up"], moe.top_k, moe.scaling, T)
        e = _k("moe_down", li, moe_down, hexp, inds, L["down"], moe.top_k, T)
        g = _k("out_cat", li, gemv, [e, hcat], [L["out_cat"]], prologue="relu2cat", Ka=moe.d_latent, sa=moe.d_latent,
               sb=hcat.shape[1], T=T, K=moe.d_latent + moe.d_shared)[0]
    else:
        parts = _k("moe", li, moe_fused, logits, L["ebias"], hcat, hcat.shape[1], L["up"], L["down"], moe.top_k,
                   moe.scaling, T)
        if _MOE_ATOMIC:
            g = _k("out_cat", li, gemv, [parts, hcat], [L["out_cat"]], prologue="relu2cat", Ka=moe.d_latent,
                   sa=moe.d_latent, sb=hcat.shape[1], T=T, K=moe.d_latent + moe.d_shared)[0]
        else:
            g = _k("out_cat", li, gemv, [parts, hcat], [L["out_cat"]], prologue="moecat", Ka=moe.d_latent,
                   sa=parts.shape[1], sb=hcat.shape[1], T=T, K=moe.d_latent + moe.d_shared)[0]
    return resid, g


def _m_layer(L, prev_li, T, D, first):
    """Whole M layer as one function (compiled per layer / batch size)."""
    H, P, N, R, A, n_static = L["dims"]
    li = L["li"]

    def f(resid, g, prev_geo, st):
        if first:
            outs = gemv([resid], [L["in"]], prologue="norm", T=T, K=D)
        else:
            outs = _k("in", li, gemv, [resid, g, prev_geo], [L["in"]], prologue="geo", LI=prev_li, T=T, K=D)
        resid = outs[-1]
        y, st = _k("m3", li, m3_step, outs[0], L["prm"]["m3"], st, H, P, N, R, A, n_static)
        gmix = _k("mixer_proj", li, gemv, [y], [L["mixer_proj"]], T=T)[0]
        resid, g = _moe_block(L, resid, gmix, T, D)
        return resid, g, st

    return f


_MOE_SPLIT = os.environ.get("OLALA_MOE_SPLIT", "0") == "1"
_compiled = {}
_COMPILE = os.environ.get("OLALA_COMPILE", "1") == "1" and not (_SKIP or _STUB)


def decode(model, x, cache, positions):
    """One fused decode step. x: (B, 1, D) f32 embeddings (not yet normalized)."""
    plan = _plan(model)
    B, _, D = x.shape
    T = B
    xin = x.reshape(T, D)
    n_layers = len(model.layers)
    shift = dict(zip(model.attn_idx, cache[n_layers:], strict=True))
    pos = positions.reshape(T).astype(mx.int32)

    kstart = None  # first valid key per row (same for every attention layer)
    resid = xin  # residual stream (T, D) f32 (embedding before the first layer)
    g = None  # pending update for the geodesic of the previous sub-layer
    prev_geo, prev_li = None, 0
    for L in plan.layers:
        li = L["li"]
        first = g is None
        if L["type"] == "M":
            key = (id(plan), li, T)
            fn = _compiled.get(key)
            if fn is None:
                fn = _m_layer(L, prev_li, T, D, first)
                if _COMPILE:
                    fn = mx.compile(fn)
                _compiled[key] = fn
            c = cache[li]
            gg = g if g is not None else xin
            pg = prev_geo if prev_geo is not None else L["geo_mlp"]
            resid, g, c[0] = fn(resid, gg, pg, c[0])
        else:
            segs = [L["in"], L["in_small"]]
            if first:
                outs = gemv([resid], segs, prologue="norm", T=T, K=D)
            else:
                outs = _k("in_v", li, gemv, [resid, g, prev_geo], segs, prologue="geo", LI=prev_li, T=T, K=D)
            resid = outs[-1]
            att = L["attn"]
            big, small = outs[0], outs[1]
            sc = shift[li]
            Hq, Hk, Dh, Rk = att.n_heads, att.n_kv, att.head_dim, att.rank
            kc = cache[li]
            kv_dtype = kc.keys.dtype
            if sc[0] is None:
                kprev = mx.zeros((T, 1, Hk, Dh), mx.float32)
                vprev = kprev
            else:
                kprev, vprev = sc[0].astype(mx.float32), sc[1].astype(mx.float32)
            q, k_new, v_new, kraw, vraw = v_prep(
                big, small, kprev, vprev, L["prm"], pos, Hq, Hk, Dh, Rk, att.n_noise, att.window, T, kv_dtype
            )
            sc[0], sc[1] = kraw, vraw
            keys, values = kc.update_and_fetch(k_new, v_new)
            if kstart is None:
                S_len = keys.shape[2]
                lp = getattr(kc, "left_padding", None)
                ks = max(S_len - 1 - att.window, 0)
                kstart = (mx.maximum(lp, ks) if lp is not None else mx.full((T,), ks)).astype(mx.int32)
            o, m, l = attn_decode(q, keys, values, kstart, att.scale, att.softcap, T, Hq, Hk, Dh)
            ymix = attn_combine(
                o, m, l, small, big, Hq, Hk, Dh, att.snr, att.n_noise, att.gate_bias, T,
                small_off=2 * Hk * Rk + 2 * Hk, big_off=Hq * Dh + 2 * Rk * Dh,
            )
            gmix = _k("mixer_proj_v", li, gemv, [ymix], [L["mixer_proj"]], T=T)[0]
            resid, g = _moe_block(L, resid, gmix, T, D)
        prev_geo, prev_li = L["geo_mlp"], li
    # final geodesic update; the caller applies the head (see head())
    return geodesic(resid, g, prev_geo, prev_li, T, D).reshape(B, 1, D)


def head(lm_head, x):
    """lm_head on f32 activations. x: (B, 1, D)."""
    B, _, D = x.shape
    _cur[0] = "head"
    return gemv([x.reshape(B, D)], [W(lm_head)], T=B)[0].reshape(B, 1, -1)
