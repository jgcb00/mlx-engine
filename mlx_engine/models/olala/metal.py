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

MAX_ROWS = 128  # fused decode up to this batch size
# Above these many rows, matrix products use MLX's (q)matmul instead of the GEMV kernels
# (measured crossovers on M3 Ultra).
GEMV_MAX_T = 8  # bf16 / f16 weights
GEMV_MAX_TQ = 5  # quantized weights


def _gemv_limit(ws):
    return GEMV_MAX_TQ if any(w.scales is not None for w in ws) else GEMV_MAX_T


NSG = 16  # simdgroups per threadgroup in the GEMV kernels (16 x 1 row beat 8 x 4 rows by 25%)
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

    def __init__(
        self, module_or_array, scales=None, biases=None, bits=None, group_size=None
    ):
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
            if self.bits not in (4, 5, 6, 8) or self.gs not in (32, 64, 128):
                raise ValueError("unsupported quantization")
            self.rows = self.w.shape[-2]
            self.cols = self.w.shape[-1] * 32 // self.bits
        else:
            self.fmt = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[
                self.w.dtype
            ]
            self.rows, self.cols = self.w.shape[-2], self.w.shape[-1]

    def arrays(self):
        if self._arrays is None:
            self._arrays = [
                _handle(a)
                for a in (
                    [self.w]
                    if self.scales is None
                    else [self.w, self.scales, self.biases]
                )
            ]
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
    return {"bf16": 8, "f16": 8, "f32": 4, "q8": 16, "q4": 16, "q6": 16, "q5": 32}[fmt]


def _unpack_code(bits, n):
    """Code turning the n * bits / 32 words wq[] of a little-endian bit stream (MLX's 5 / 6 bit
    packing) into float4 f[n / 4]."""
    mask = (1 << bits) - 1
    vals = []
    for i in range(n):
        st = bits * i
        w, off = divmod(st, 32)
        if off + bits <= 32:
            vals.append(f"((wq[{w}] >> {off}) & {mask}u)")
        else:
            vals.append(
                f"(((wq[{w}] >> {off}) | (wq[{w + 1}] << {32 - off})) & {mask}u)"
            )
    return "\n".join(
        f"f[{j}] = float4({vals[4 * j]}, {vals[4 * j + 1]}, {vals[4 * j + 2]}, {vals[4 * j + 3]});"
        for j in range(n // 4)
    )


def _dot_code(fmt, name, K, gs, xexpr, tdim, rowvar, kvar):
    """Code accumulating acc[r][t] += dot(W[row, k:k+chunk], x[t, k:k+chunk]) for one 16-byte chunk.
    xexpr(t, j) must give a float4 of x for token t at k + 4*j."""
    lines = []
    if fmt in ("bf16", "f16"):
        vt = "bfloat4" if fmt == "bf16" else "half4"
        lines.append(
            f"const device {vt}* wp = (const device {vt}*)({name} + (size_t){rowvar} * {K} + {kvar});"
        )
        lines.append("const float4 wv0 = float4(wp[0]), wv1 = float4(wp[1]);")
        lines.append(
            f"for (int t = 0; t < {tdim}; ++t) acc[r][t] += dot(wv0, {xexpr('t', 0)}) + dot(wv1, {xexpr('t', 1)});"
        )
    elif fmt == "f32":
        lines.append(
            f"const float4 wv0 = *(const device float4*)({name} + (size_t){rowvar} * {K} + {kvar});"
        )
        lines.append(
            f"for (int t = 0; t < {tdim}; ++t) acc[r][t] += dot(wv0, {xexpr('t', 0)});"
        )
    elif fmt == "q8":
        lines.append(
            f"const uint4 q = *(const device uint4*)({name} + (size_t){rowvar} * ({K} / 4) + {kvar} / 4);"
        )
        lines.append(
            f"const size_t gi = (size_t){rowvar} * ({K} / {gs}) + {kvar} / {gs};"
        )
        lines.append(f"const float sc = float({name}_s[gi]), bi = float({name}_b[gi]);")
        lines.append("float4 f[4];")
        lines.append(
            "for (int j = 0; j < 4; ++j) { const uint u = q[j]; f[j] = float4(u & 0xffu, (u >> 8) & 0xffu, (u >> 16) & 0xffu, u >> 24); }"
        )
        if (
            tdim > 1
        ):  # several tokens share the chunk: dequantize once, then plain dot products
            lines.append("for (int j = 0; j < 4; ++j) f[j] = f[j] * sc + bi;")
            lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0;")
            lines.append(
                f"  for (int j = 0; j < 4; ++j) s1 += dot(f[j], {xexpr('t', 'j')});"
            )
            lines.append("  acc[r][t] += s1; }")
            return "\n".join(lines)
        lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0, s2 = 0;")
        lines.append(
            f"  for (int j = 0; j < 4; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}"
        )
        lines.append("  acc[r][t] += sc * s1 + bi * s2; }")
    elif fmt in ("q5", "q6"):
        bits = int(fmt[1])
        n = _chunk(fmt)
        nw = n * bits // 32
        lines.append(
            f"const device uint* qp = (const device uint*)({name}) + (size_t){rowvar} * ({K} * {bits} / 32) + {kvar} * {bits} / 32;"
        )
        lines.append(f"uint wq[{nw}]; for (int i = 0; i < {nw}; ++i) wq[i] = qp[i];")
        lines.append(
            f"const size_t gi = (size_t){rowvar} * ({K} / {gs}) + {kvar} / {gs};"
        )
        lines.append(f"const float sc = float({name}_s[gi]), bi = float({name}_b[gi]);")
        lines.append(f"float4 f[{n // 4}];")
        lines.append(_unpack_code(bits, n))
        if tdim > 1:
            lines.append(f"for (int j = 0; j < {n // 4}; ++j) f[j] = f[j] * sc + bi;")
            lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0;")
            lines.append(
                f"  for (int j = 0; j < {n // 4}; ++j) s1 += dot(f[j], {xexpr('t', 'j')});"
            )
            lines.append("  acc[r][t] += s1; }")
            return "\n".join(lines)
        lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0, s2 = 0;")
        lines.append(
            f"  for (int j = 0; j < {n // 4}; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}"
        )
        lines.append("  acc[r][t] += sc * s1 + bi * s2; }")
    elif fmt == "q4":
        # 16 values per 8-byte load: 3 balanced chunks per lane at K = 1536
        lines.append(
            f"const uint2 q = *(const device uint2*)({name} + (size_t){rowvar} * ({K} / 8) + {kvar} / 8);"
        )
        lines.append(
            f"const size_t gi = (size_t){rowvar} * ({K} / {gs}) + {kvar} / {gs};"
        )
        lines.append(f"const float sc = float({name}_s[gi]), bi = float({name}_b[gi]);")
        lines.append("float4 f[4];")
        lines.append(
            "for (int j = 0; j < 2; ++j) { const uint u = q[j]; "
            "f[2*j] = float4(u & 0xfu, (u >> 4) & 0xfu, (u >> 8) & 0xfu, (u >> 12) & 0xfu); "
            "f[2*j+1] = float4((u >> 16) & 0xfu, (u >> 20) & 0xfu, (u >> 24) & 0xfu, u >> 28); }"
        )
        if tdim > 1:
            lines.append("for (int j = 0; j < 4; ++j) f[j] = f[j] * sc + bi;")
            lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0;")
            lines.append(
                f"  for (int j = 0; j < 4; ++j) s1 += dot(f[j], {xexpr('t', 'j')});"
            )
            lines.append("  acc[r][t] += s1; }")
            return "\n".join(lines)
        lines.append(f"for (int t = 0; t < {tdim}; ++t) {{ float s1 = 0, s2 = 0;")
        lines.append(
            f"  for (int j = 0; j < 4; ++j) {{ const float4 xv = {xexpr('t', 'j')}; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }}"
        )
        lines.append("  acc[r][t] += sc * s1 + bi * s2; }")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# GEMV with fused prologue
# --------------------------------------------------------------------------

_kernels = {}


def _geo_prologue(K, T, LI):
    """xs[t] = geodesic(xin[t], g[t]); also written to xout by threadgroup 0.
    x and g are read once into registers (EPT elements per thread)."""
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
        theta = min((theta * scale + gbias) / geo[2], {math.pi / 4}f);
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


def _geos_prologue(K, T):
    """xs[t] = geodesic(xin[t], g[t]) from the sums gst[t] = (x.x, x.g, g.g) accumulated by the
    GEMV that produced g (see _stats_epilogue): no reductions, one barrier."""
    NT = NSG * 32
    return f"""
    {{
    const float scale = geo[0], gbias = geo[1], lidx = geo[2];
    for (int t = 0; t < TN; ++t) {{
        const float xx = max(gst[4 * t], 1e-12f), xg = gst[4 * t + 1], gg = gst[4 * t + 2];
        const float proj = xg / xx;
        const float tn = max(sqrt(max(gg - 2.0f * proj * xg + proj * proj * xx, 0.0f)), 1e-8f);
        const float rr = max(sqrt(xx), 1e-6f);
        float theta = min(tn / rr, {math.pi / 4}f);
        theta = min((theta * scale + gbias) / lidx, {math.pi / 4}f);
        const float c = cos(theta), s = sin(theta) * rr / tn;
        for (uint i = tid; i < {K}; i += {NT}) {{
            const float a = xin[t * {K} + i];
            const float v = a * c + (g[t * {K} + i] - proj * a) * s;
            xs[t * {K} + i] = v;
            if (tgid == 0) xout[t * {K} + i] = v;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    """


def _stats_epilogue(T):
    """Adds this threadgroup's (x.x, x.g, g.g) partial sums of its rows to stats_in (atomic)."""
    return f"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)TN * 3) {{
        const int t = tid / 3, c = tid % 3;
        float acc_s = 0;
        for (int i = 0; i < {NSG}; ++i) acc_s += gs_part[i][t][c];
        atomic_fetch_add_explicit((device atomic<float>*)stats_in + 4 * t + c, acc_s, memory_order_relaxed);
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


_tcounts = {}


def _tcount(T):
    a = _tcounts.get(T)
    if a is None:
        a = _tcounts[T] = mx.array([T, 0, 0, 0], dtype=mx.int32)
        mx.eval(a)
    return a


def _row_bucket(T):
    """GEMV kernels are compiled for a maximum row count and get the actual one at run time,
    so a new batch size does not trigger a recompilation of every kernel."""
    return 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8


def gemv_kernel(
    K,
    T,
    segs,
    prologue,
    LI=0,
    Ka=0,
    sa=0,
    sb=0,
    x_in_tg=True,
    RPS=4,
    ndeps=0,
    stats=False,
):
    """segs: list of (fmt, gs, rows). Output j is (rows_valid, rows_j) f32. T is the compile-time
    maximum number of rows (see _row_bucket); the actual count TN comes from the input tcnt.
    ndeps extra (unused) inputs order the kernel after the kernels that produced them."""
    # the layer index of a geodesic prologue is read from geo[2] at run time
    key = ("gemv", K, T, tuple(segs), prologue, Ka, sa, sb, x_in_tg, RPS, ndeps, stats)
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
    elif prologue == "geos":
        inputs += ["xin", "g", "geo", "gst"]
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
    inputs += [f"dep{j}" for j in range(ndeps)]
    inputs.append("tcnt")
    if stats:
        inputs += ["xres", "stats_in"]
    outputs = [f"y{j}" for j in range(len(segs))]
    if prologue in ("geo", "geos", "norm"):
        outputs.append("xout")

    if prologue == "relu2cat":
        decl = ""

        def xexpr(t, j):
            return (
                f"((k + 4 * ({j}) < {Ka}) ? *(const device float4*)(a + min({t}, TN - 1) * {sa} + k + 4 * ({j})) : "
                f"relu2(*(const device float4*)(b + min({t}, TN - 1) * {sb} + k + 4 * ({j}))))"
            )
    elif x_in_tg:
        decl = f"threadgroup float xs[{T} * {K}];"

        def xexpr(t, j):
            return f"*(const threadgroup float4*)(xs + {t} * {K} + k + 4 * ({j}))"
    else:
        decl = ""

        def xexpr(t, j):
            return (
                f"*(const device float4*)(x + min({t}, TN - 1) * {K} + k + 4 * ({j}))"
            )

    pro = {
        "geo": lambda: _geo_prologue(K, T, LI).replace(
            f"for (int t = 0; t < {T}; ++t)", "for (int t = 0; t < TN; ++t)"
        ),
        "norm": lambda: _norm_prologue(K, T).replace(
            f"for (int t = 0; t < {T}; ++t)", "for (int t = 0; t < TN; ++t)"
        ),
        "geos": lambda: _geos_prologue(K, T),
        "relu2cat": lambda: "",
        "moecat": lambda: (
            f"""
    for (uint i = tid; i < TN * {K}; i += {NSG} * 32) {{
        const uint t = i / {K}, k = i % {K};
        float v;
        if (k < {Ka}) {{ v = 0; for (int p = 0; p < {sa}; ++p) v += parts[(t * {sa} + p) * {Ka} + k]; }}
        else {{ const float u = max(b[t * {sb} + k], 0.0f); v = u * u; }}
        xs[i] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """
        ),
        "plain": lambda: (
            f"for (uint i = tid; i < TN * {K}; i += {NSG} * 32) xs[i] = x[i];\n"
            "threadgroup_barrier(mem_flags::mem_threadgroup);"
            if x_in_tg
            else ""
        ),
    }[prologue]()

    total = sum(s[2] for s in segs)
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
        outs.append(
            f"if (row >= {off} && row < {off + rows}) y{j}[t * {rows} + row - {off}] = v;"
        )
        off += rows
    outs = "\n".join(outs)

    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint tgid = threadgroup_position_in_grid.x;
    const int TN = tcnt[0];
    {decl}
    {pro}
    const int row0 = (tgid * {NSG} + sg) * {RPS};
    float acc[{RPS}][{T}];
    for (int r = 0; r < {RPS}; ++r) for (int t = 0; t < {T}; ++t) acc[r][t] = 0;
    for (int r = 0; r < {RPS}; ++r) {{
        const int row = row0 + r;
        {body}
    }}
    {"threadgroup float gs_part[" + str(NSG) + "][" + str(T) + "][3];" if stats else ""}
    for (int r = 0; r < {RPS}; ++r) {{
        const int row = row0 + r;
        if (row >= {total}) {{
            {"if (lane == 0) for (int t = 0; t < " + str(T) + "; ++t) { gs_part[sg][t][0] = 0; gs_part[sg][t][1] = 0; gs_part[sg][t][2] = 0; }" if stats else ""}
            break;
        }}
        for (int t = 0; t < TN; ++t) {{
            const float v = simd_sum(acc[r][t]);
            if (lane == 0) {{ {outs}
                {"const float xr = xres[t * " + str(total) + " + row]; gs_part[sg][t][0] = xr * xr; gs_part[sg][t][1] = xr * v; gs_part[sg][t][2] = v * v;" if stats else ""}
            }}
        }}
    }}
    {_stats_epilogue(T) if stats else ""}
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


def mlx_linear(x, w: "W"):
    """x (T, K) f32 @ W.T with MLX kernels; f32 output. Quantized weights take f32 activations
    directly; bf16 / f16 weights take activations in the weight dtype."""
    if w.scales is not None:
        return mx.quantized_matmul(
            x, w.w, w.scales, w.biases, transpose=True, group_size=w.gs, bits=w.bits
        )
    if w.w.dtype == mx.float32:
        return x @ w.w.T
    return (x.astype(w.w.dtype) @ w.w.T).astype(mx.float32)


def _gemv_mlx(x_inputs, ws, prologue, LI, Ka, sa, sb, T, K):
    """Many rows: prologue as its own kernel / ops, products through MLX (q)matmul."""
    extra = []
    if prologue == "geo":
        x = geodesic(x_inputs[0], x_inputs[1], x_inputs[2], LI, T, K)
        extra = [x]
    elif prologue == "norm":
        x0 = x_inputs[0]
        x = x0 / mx.maximum(mx.sqrt(mx.sum(x0 * x0, axis=-1, keepdims=True)), 1e-12)
        extra = [x]
    elif prologue == "relu2cat":
        a, b = x_inputs
        x = mx.concatenate([a[:, :Ka], mx.square(nn.relu(b[:, Ka:K]))], axis=-1)
    elif prologue == "moecat":
        parts, b = x_inputs
        x = mx.concatenate([parts.sum(axis=1), mx.square(nn.relu(b[:, Ka:K]))], axis=-1)
    else:
        x = x_inputs[0]
    return [mlx_linear(x, w) for w in ws] + extra


def gemv(
    x_inputs,
    ws,
    prologue="plain",
    LI=0,
    Ka=0,
    sa=0,
    sb=0,
    T=1,
    K=None,
    RPS=None,
    deps=(),
    stats_x=None,
):
    """Run a fused GEMV. x_inputs: list of input arrays for the prologue. ws: list of W.
    Returns [y_j (T, rows_j) f32 ...] (+ xout (T, K) for geo / geos / norm).

    stats_x: the residual the next geodesic will rotate; the GEMV then also returns, last,
    the (T, 4) sums (x.x, x.g, g.g) with g its output (prologue "geos" of the consumer), or
    None when it cannot (the consumer then uses "geo")."""
    RPS = RPS or 1
    K = K or ws[0].cols
    segs = [(w.fmt, w.gs or 0, w.rows) for w in ws]
    if prologue == "geos" and (T > _gemv_limit(ws) or T * K * 4 > TG_X_BYTES):
        prologue, x_inputs = "geo", x_inputs[:3]
    if T > _gemv_limit(ws):
        out = _gemv_mlx(x_inputs, ws, prologue, LI, Ka, sa, sb, T, K)
        return out + [None] if stats_x is not None else out
        return _gemv_mlx(x_inputs, ws, prologue, LI, Ka, sa, sb, T, K)
    if prologue in ("geo", "norm") and T * K * 4 > TG_X_BYTES:
        # too wide for threadgroup memory: run the prologue on its own, then a plain GEMV
        if prologue == "geo":
            xs = geodesic(x_inputs[0], x_inputs[1], x_inputs[2], LI, T, K)
        else:
            x = x_inputs[0]
            xs = x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)
        out = gemv([xs], ws, T=T, K=K, RPS=RPS) + [xs]
        return out + [None] if stats_x is not None else out
    if prologue == "moecat" and T * K * 4 > TG_X_BYTES:
        parts, b = x_inputs
        e = parts.sum(axis=1)  # (T, Ka)
        return gemv(
            [e, b],
            ws,
            prologue="relu2cat",
            Ka=Ka,
            sa=Ka,
            sb=sb,
            T=T,
            K=K,
            RPS=RPS,
            stats_x=stats_x,
        )
    TC = _row_bucket(T)
    if prologue in ("geo", "geos", "norm", "moecat") and TC * K * 4 > TG_X_BYTES:
        TC = T  # exact size when the bucket would not fit in threadgroup memory
    # computed inputs are staged in threadgroup memory; plain inputs are read from device
    # memory (faster than staging them, measured)
    x_in_tg = prologue in ("geo", "geos", "norm", "moecat")
    stats = stats_x is not None
    kern, total, _ = gemv_kernel(
        K,
        TC,
        segs,
        prologue,
        LI,
        Ka,
        sa,
        sb,
        x_in_tg,
        RPS,
        ndeps=len(deps),
        stats=stats,
    )
    ntg = (total + NSG * RPS - 1) // (NSG * RPS)
    inputs = list(x_inputs)
    for w in ws:
        inputs += w.arrays()
    inputs += [d.reshape(-1)[:4] for d in deps]
    inputs.append(_tcount(T))
    if stats:
        # padded: MLX binds very small inputs as `constant` buffers, which cannot be written
        gst = zeros_f32((T * 4 + 64,))
        inputs += [stats_x, gst]
    shapes = [(T, w.rows) for w in ws]
    dtypes = [mx.float32] * len(ws)
    if prologue in ("geo", "geos", "norm"):
        shapes.append((T, K))
        dtypes.append(mx.float32)
    if stats:
        return kern(
            inputs=inputs,
            grid=(ntg * NSG * 32, 1, 1),
            threadgroup=(NSG * 32, 1, 1),
            output_shapes=shapes,
            output_dtypes=dtypes,
        ) + [gst]
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


def geo_params(module, layer_idx):
    """[scale, bias, layer_idx + 1] (f32) of a geodesic update, as read by the kernels."""
    return mx.concatenate(
        [
            module.scale.astype(mx.float32).reshape(1),
            module.bias.astype(mx.float32).reshape(1),
            mx.array([layer_idx + 1.0], dtype=mx.float32),
        ]
    )


def geodesic(xin, g, geo, LI, T, K):
    """Standalone geodesic update, one threadgroup per row."""
    key = ("geo1", K)
    if key not in _kernels:
        body = (
            _geo_prologue(K, 1, LI)
            .replace("xin[t * " + str(K), "xin[row * " + str(K))
            .replace("g[t * " + str(K), "g[row * " + str(K))
            .replace("xs[t * " + str(K) + " + i] = v;", "")
            .replace("xout[t * " + str(K), "xout[row * " + str(K))
            .replace("if (tgid == 0) xout", "xout")
        )
        src = (
            """
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint row = threadgroup_position_in_grid.x;
        const uint tgid = 0;
        """
            + body
        )
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_geo1_{K}",
            input_names=["xin", "g", "geo"],
            output_names=["xout"],
            source=src,
            header=_HDR,
        )
    return _kernels[key](
        inputs=[xin, g, geo],
        grid=(T * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(T, K)],
        output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# Mamba-3 MIMO decode step
# --------------------------------------------------------------------------

M3_PB = 64  # state rows (p) per threadgroup in m3_step (one threadgroup per head)


def m3_param_pack(m):
    """f32 [dt_bias (H) | D (H) | b_norm (N) | c_norm (N) | b_bias | c_bias (H*R*N) | mimo_x | mimo_z | mimo_o (H*R*P)]."""
    f32 = mx.float32
    bn = m.B_norm.norm.weight.astype(f32)
    cn = m.C_norm.norm.weight.astype(f32)
    if m.zc:
        bn, cn = 1.0 + bn, 1.0 + cn
    parts = [
        m.dt_bias,
        m.D,
        bn,
        cn,
        m.B_bias,
        m.C_bias,
        m.in_proj_mimo_x,
        m.in_proj_mimo_z,
        m.out_proj_mimo,
    ]
    return mx.concatenate([a.astype(f32).reshape(-1) for a in parts])


def _m3_proj_code(wmp, K, PB):
    """gmix[b, o] += sum_p W[o, h*P + pb*PB + p] * ys[p] for this threadgroup's rows (atomic)."""
    ch = _chunk(wmp.fmt)
    assert PB % ch == 0

    def xexpr(t, j):
        return f"*(const threadgroup float4*)(ys + (k - c0) + 4 * ({j}))"

    dot = _dot_code(wmp.fmt, "wm", K, wmp.gs or 0, xexpr, 1, "o", "k").replace(
        "acc[r][t]", "acc"
    )
    return f"""
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {{
            device atomic<float>* gm = (device atomic<float>*)gmix_in;
            const int c0 = h * P + pb * PB;
            for (int o = tid; o < {wmp.rows}; o += NSGM * 32) {{
                float acc = 0;
                for (int k = c0; k < c0 + PB; k += {ch}) {{
                    {dot}
                }}
                atomic_fetch_add_explicit(&gm[(size_t)b * {wmp.rows} + o], acc, memory_order_relaxed);
            }}
        }}
    """


def zeros_f32(shape):
    """A materialized zero (T, D) f32 buffer (kernels accumulate into it atomically)."""
    key = ("zeros",)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name="olala_zeros",
            input_names=["n_"],
            output_names=["z"],
            source="const uint i = thread_position_in_grid.x; if (i < (uint)n_[0]) z[i] = 0.0f;",
        )
    n = 1
    for d in shape:
        n *= d
    return _kernels[key](
        inputs=[mx.array([n, 0, 0, 0], dtype=mx.int32)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )[0]


def m3_step(proj, prm, st, H, P, N, R, A, n_static, wmp=None, gmix=None):
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
    fused = wmp is not None
    key = (
        "m3v2",
        H,
        P,
        N,
        R,
        A,
        n_static,
        PB,
        wmp.key() if fused else None,
        wmp.rows if fused else 0,
    )
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
            if (lane == 0) {{ {"ys[pl] = acc + Dh * d;" if fused else "y[(size_t)b * H * P + h * P + p] = acc + Dh * d;"} }}
        }}
        {_m3_proj_code(wmp, H * P, PB) if fused else ""}
        """
        if fused:
            src = src.replace(
                "threadgroup float vws[3][R][PB];",
                "threadgroup float vws[3][R][PB];\n        threadgroup float ys[PB];",
            )
            names = (
                ["proj", "prm", "st", "wm"]
                + (["wm_s", "wm_b"] if wmp.fmt.startswith("q") else [])
                + ["gmix_in"]
            )
            _kernels[key] = mx.fast.metal_kernel(
                name=f"olala_m3v2f_{abs(hash(key)) % 10**12}",
                input_names=names,
                output_names=["st_out"],
                source=src,
                header=_HDR,
            )
        else:
            _kernels[key] = mx.fast.metal_kernel(
                name=f"olala_m3v2_{PB}",
                input_names=["proj", "prm", "st"],
                output_names=["y", "st_out"],
                source=src,
                header=_HDR,
            )
    kern = _kernels[key]
    npb = P // PB
    if fused:
        return kern(
            inputs=[proj, prm, st] + wmp.arrays() + [gmix],
            grid=(T * H * npb * NSG_ * 32, 1, 1),
            threadgroup=(NSG_ * 32, 1, 1),
            output_shapes=[(T, H, SW)],
            output_dtypes=[mx.float32],
        )[0]
    return kern(
        inputs=[proj, prm, st],
        grid=(T * H * npb * NSG_ * 32, 1, 1),
        threadgroup=(NSG_ * 32, 1, 1),
        output_shapes=[(T, H * P), (T, H, SW)],
        output_dtypes=[mx.float32] * 2,
    )


M3P_PB = 64  # state rows per threadgroup in m3_prefill


def m3_prefill(proj, prm, st, H, P, N, R, A, n_static, rng=None, seg=1, write_y=True):
    """Sequential Mamba-3 MIMO over a prompt, fused (prep + state sweep per token).

    proj: (B, L, n_static + n_dyn) f32; prm: m3_param_pack(); st: packed state (B, H, SW).
    rng: optional (B, 2) int32 valid token range [s0, s1) per row (padding).
    Returns y (B, L, H*P) f32 and the new packed state.

    Threadgroup = (row, head, block of PB state rows), 16 simdgroups, one state row per
    simdgroup and lane = 4 consecutive state columns, state kept in registers across tokens.
    Simdgroups 0..2R-1 normalize / bias / rotate one row of [B | C] per token (lane = rotary
    angle, carried in a register); the others build v and w for the block. Rotated keys /
    queries and v / w are triple-buffered in threadgroup memory: one barrier per token."""
    B, L, _PW = proj.shape
    SW = P * N + R * N + P + A
    PB = M3P_PB
    NSG_ = 16
    assert N == 128 and A == 32 and 2 * R <= NSG_ and PB <= NSG_ * 4
    if rng is None:
        rng = mx.broadcast_to(mx.array([0, L], dtype=mx.int32)[None], (B, 2))
    key = ("m3p", H, P, N, R, A, n_static, PB, seg, write_y)
    if key not in _kernels:
        RPSG = (PB + NSG_ - 1) // NSG_
        src = f"""
        constexpr int H = {H}, P = {P}, N = {N}, R = {R}, A = {A}, PB = {PB}, NSGM = {NSG_}, RPSG = {RPSG};
        constexpr int PH = 2 * P + 3, NS = {n_static}, PW = {n_static} + 2 * R * N + A;
        constexpr int SW = {SW}, OK_ = P * N, OX = OK_ + R * N, OA = OX + P;
        constexpr int O_DT = 0, O_D = H, O_BN = 2 * H, O_CN = O_BN + N, O_BB = O_CN + N, O_CB = O_BB + H * R * N;
        constexpr int O_MX = O_CB + H * R * N, O_MZ = O_MX + H * R * P, O_MO = O_MZ + H * R * P;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int npb = P / PB;
        const int b = blk / (H * npb);
        const int h = (blk / npb) % H;
        const int pb = blk % npb;
        const int L = proj_shape[1];
        const int s0 = rng[2 * b], s1 = rng[2 * b + 1];
        constexpr int SEG = {seg};   // rows per batch entry (segments of the prompt)
        const device float* pr0 = proj + (size_t)(b / SEG) * L * PW;
        float lsum = 0;              // sum of log(alpha) over the range
        const device float* sin_ = st + ((size_t)b * H + h) * SW;
        device float* sout = st_out + ((size_t)b * H + h) * SW;

        threadgroup float4 kqs[3][2 * R][N / 4];
        threadgroup float vws[3][2][R][PB];      // [buf][v | w][r][row]

        // constant per-head parameters
        const float dtb = prm[O_DT + h], Dh = prm[O_D + h];
        const bool kq_sg = sg < 2 * R;
        const int gi = sg;
        const bool isq = gi >= R;
        const int rr = isq ? gi - R : gi;
        float4 wn = 0, bb = 0;
        if (kq_sg) {{
            wn = ((const device float4*)(prm + (isq ? O_CN : O_BN)))[lane];
            bb = ((const device float4*)(prm + (isq ? O_CB : O_BB) + (h * R + rr) * N))[lane];
        }}
        float th = sin_[OA + lane];                      // rotary angle of this lane (A == 32)
        // state rows of this simdgroup
        float4 srow[RPSG];
        for (int i = 0; i < RPSG; ++i) {{
            const int pl = sg + NSGM * i;
            srow[i] = pl < PB ? ((const device float4*)(sin_ + (pb * PB + pl) * N))[lane] : 0;
        }}
        // seed buffer (s0 - 1) % 3 with the previous token: rotated k from the state, v from x
        {{
            const int pbuf = (s0 + 2) % 3;
            if (sg < R) kqs[pbuf][sg][lane] = ((const device float4*)(sin_ + OK_ + sg * N))[lane];
            for (int i = tid; i < R * PB; i += NSGM * 32) {{
                const int r = i / PB, pl = i % PB, p = pb * PB + pl;
                vws[pbuf][0][r][pl] = sin_[OX + p] * prm[O_MX + (h * R + r) * P + p];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // kv of the token before the range (from the seeded buffer)
        float4 kvprev[RPSG];
        {{
            const int sb0 = (s0 + 2) % 3;
            for (int i = 0; i < RPSG; ++i) {{
                const int pl = sg + NSGM * i;
                float4 a0 = 0;
                if (pl < PB) for (int r = 0; r < R; ++r) a0 += vws[sb0][0][r][pl] * kqs[sb0][r][lane];
                kvprev[i] = a0;
            }}
        }}
        // per-thread constants for the v / w builders (fixed (r, row) items) and prefetched inputs
        constexpr int VW_T = (NSGM - 2 * R) * 32;
        constexpr int VW_N = (R * PB + VW_T - 1) / VW_T;
        float cmx[VW_N], cmz[VW_N], cmo[VW_N];
        int cp[VW_N], cr[VW_N], cpl[VW_N];
        if (!kq_sg) {{
            for (int j = 0; j < VW_N; ++j) {{
                const int i = (tid - 2 * R * 32) + j * VW_T;
                const int r = min(i, R * PB - 1) / PB, pl = min(i, R * PB - 1) % PB, p = pb * PB + pl;
                cr[j] = i < R * PB ? r : -1; cpl[j] = pl; cp[j] = p;
                cmx[j] = prm[O_MX + (h * R + r) * P + p];
                cmz[j] = prm[O_MZ + (h * R + r) * P + p];
                cmo[j] = prm[O_MO + (h * R + r) * P + p];
            }}
        }}
        // prefetch registers for token t
        float3 sc_n = 0; float ang_n = 0; float4 u_n = 0; float xz_n[VW_N][2];
        auto fetch = [&](int t) {{
            const device float* pr = pr0 + (size_t)t * PW;
            const device float* ph = pr + h * PH;
            sc_n = float3(ph[2 * P], ph[2 * P + 1], ph[2 * P + 2]);
            ang_n = pr[NS + 2 * R * N + lane];
            if (kq_sg) u_n = ((const device float4*)(pr + NS + gi * N))[lane];
            else for (int j = 0; j < VW_N; ++j) {{ xz_n[j][0] = ph[P + cp[j]]; xz_n[j][1] = ph[cp[j]]; }}
        }};
        if (s1 > s0) fetch(s0);
        for (int t = s0; t < s1; ++t) {{
            const int cb = t % 3, pbuf = (t + 2) % 3;
            const float3 scv = sc_n; const float angv = ang_n; const float4 u = u_n;
            float xz[VW_N][2];
            for (int j = 0; j < VW_N; ++j) {{ xz[j][0] = xz_n[j][0]; xz[j][1] = xz_n[j][1]; }}
            if (t + 1 < s1) fetch(t + 1);
            const float dt = olala_softplus(scv.x + dtb);
            const float a = min(-olala_softplus(scv.y), -1e-4f);
            const float alpha = exp(a * dt);
            lsum += a * dt;
            const float trap = olala_sigmoid(scv.z);
            const float beta = (1.0f - trap) * dt * alpha;
            const float gamma = trap * dt;
            th = th + tanh(angv) * {math.pi}f * dt;
            th = th - {2 * math.pi}f * rint(th * {0.5 / math.pi}f);
            if (kq_sg) {{
                const float ss = simd_sum(dot(u, u));
                float4 val = u * rsqrt(ss / N + {1e-6}f) * wn + bb;
                const float4 partner = simd_shuffle_xor(val, 16);
                const int base = 4 * (lane & 15);
                const float4 th4 = float4(simd_shuffle(th, (base + 0) & 31), simd_shuffle(th, (base + 1) & 31),
                                          simd_shuffle(th, (base + 2) & 31), simd_shuffle(th, (base + 3) & 31));
                if ((lane & 15) < 8) {{
                    const float4 c = cos(th4), s = sin(th4);
                    val = (lane < 16) ? (val * c - partner * s) : (partner * s + val * c);
                }}
                kqs[cb][gi][lane] = val;
            }} else {{
                for (int j = 0; j < VW_N; ++j) {{
                    if (cr[j] < 0) continue;
                    const float zz = xz[j][1] * cmz[j];
                    vws[cb][0][cr[j]][cpl[j]] = xz[j][0] * cmx[j];
                    vws[cb][1][cr[j]][cpl[j]] = zz / (1.0f + exp(-zz)) * cmo[j];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 kr[R], qr[R];
            for (int r = 0; r < R; ++r) {{ kr[r] = kqs[cb][r][lane]; qr[r] = kqs[cb][R + r][lane]; }}
            for (int i = 0; i < RPSG; ++i) {{
                const int pl = sg + NSGM * i;
                if (pl >= PB) break;
                float4 kv = 0, qw = 0;
                float d = 0;
                for (int r = 0; r < R; ++r) {{
                    const float vv = vws[cb][0][r][pl], ww = vws[cb][1][r][pl];
                    kv += vv * kr[r];
                    qw += ww * qr[r];
                    d += ww * vv;
                }}
                // the previous token's outer-product term is this row's kv of the last step
                srow[i] = alpha * srow[i] + beta * kvprev[i] + gamma * kv;
                kvprev[i] = kv;
                const float acc = simd_sum(dot(srow[i], qw));
                {"if (lane == 0) y[((size_t)(b / SEG) * L + t) * H * P + h * P + pb * PB + pl] = acc + Dh * d;" if write_y else "(void)acc;"}
            }}
        }}
        if (pb == 0 && tid == 0) logA[b * H + h] = lsum;
        // new state: S rows, x of the last token, rotated k of the last token, angles
        for (int i = 0; i < RPSG; ++i) {{
            const int pl = sg + NSGM * i;
            if (pl < PB) ((device float4*)(sout + (pb * PB + pl) * N))[lane] = srow[i];
        }}
        const int lastb = (s1 + 2) % 3;
        if (s1 > s0) {{
            for (int i = tid; i < PB; i += NSGM * 32) sout[OX + pb * PB + i] = pr0[(size_t)(s1 - 1) * PW + h * PH + P + pb * PB + i];
            if (pb == 0) {{
                if (sg < R) ((device float4*)(sout + OK_ + sg * N))[lane] = kqs[lastb][sg][lane];
                if (sg == 0) sout[OA + lane] = th;
            }}
        }} else {{
            for (int i = tid; i < PB; i += NSGM * 32) sout[OX + pb * PB + i] = sin_[OX + pb * PB + i];
            if (pb == 0) {{
                if (sg < R) ((device float4*)(sout + OK_ + sg * N))[lane] = ((const device float4*)(sin_ + OK_ + sg * N))[lane];
                if (sg == 0) sout[OA + lane] = th;
            }}
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_m3p_{PB}_{seg}_{int(write_y)}",
            input_names=["proj", "prm", "st", "rng"],
            output_names=["y", "st_out", "logA"],
            source=src,
            header=_HDR,
        )
    npb = P // PB
    rows = B * seg
    y, st_out, log_a = _kernels[key](
        inputs=[proj, prm, st, rng],
        grid=(rows * H * npb * NSG_ * 32, 1, 1),
        threadgroup=(NSG_ * 32, 1, 1),
        output_shapes=[(B, L, H * P) if write_y else (1,), (rows, H, SW), (rows, H)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        init_value=0,
    )
    if seg == 1:
        return y, st_out
    return y, st_out, log_a


M3C_T = 4  # tokens per chunk in m3_chunked
M3C_PS = 1  # threadgroups per head (each owns P / M3C_PS state rows; 2 was slower)

# element (row, col) of a simdgroup_float8x8 held in thread_elements()[k] is (_FM, _FN + k)
_FRAG = """
        const short fq_ = lane / 4;
        const short _FM = (fq_ & 4) + ((lane / 2) % 4);
        const short _FN = (fq_ & 2) * 2 + (lane % 2) * 2;
"""


def _m3c_prep(proj, prm, st, rng, H, P, N, R, A, n_static):
    """Per-token inputs of m3_chunked that do not depend on the state, in parallel:
    sc (B, L, H, 4) = (log alpha, gamma, delta, 0) (zero outside the valid range),
    cs (B, L, H, 2, A) = cos / sin of the cumulative rotary angles, th (B, H, A) the final
    angles, bc (B, L, 2R, N) the normalized B | C rows (norm weight applied, no bias).
    Simdgroups [0, B*H) scan the angles of one head, the others normalize one row each."""
    B, L, _PW = proj.shape
    key = ("m3c_prep", H, P, N, R, A, n_static)
    if key not in _kernels:
        SW = P * N + R * N + P + A
        src = f"""
        constexpr int H = {H}, P = {P}, N = {N}, R = {R}, A = {A};
        constexpr int PH = 2 * P + 3, NS = {n_static}, PW = {n_static} + 2 * R * N + A;
        constexpr int SW = {SW}, OA = P * N + R * N + P;
        constexpr int O_DT = 0, O_BN = 2 * H, O_CN = O_BN + N;
        const uint lane = thread_position_in_threadgroup.x % 32;
        const int blk = threadgroup_position_in_grid.x * 8 + thread_position_in_threadgroup.x / 32;  // simdgroup
        const int L = proj_shape[1];
        const int B = proj_shape[0];
        if (blk >= B * H + B * L * 2 * R) return;
        if (blk < B * H) {{
            // blocks of 32 tokens, lane = token: scalars in parallel, angles by prefix sums
            const int b = blk / H, h = blk % H;
            const int s0 = rng[2 * b], s1 = rng[2 * b + 1];
            const device float* pr0 = proj + (size_t)b * L * PW;
            const float dtb = prm[O_DT + h];
            float carry[A];
            for (int a = 0; a < A; ++a) carry[a] = st[((size_t)b * H + h) * SW + OA + a];
            for (int t0 = 0; t0 < L; t0 += 32) {{
                const int t = t0 + lane, tl = min(t, L - 1);
                const device float* prt = pr0 + (size_t)tl * PW;
                const float3 raw = float3(prt[h * PH + 2 * P], prt[h * PH + 2 * P + 1], prt[h * PH + 2 * P + 2]);
                float4 ang[A / 4];
                for (int i = 0; i < A / 4; ++i) ang[i] = ((const device float4*)(prt + NS + 2 * R * N))[i];
                const bool ok = t >= s0 && t < s1;
                const float dt = ok ? olala_softplus(raw.x + dtb) : 0.0f;
                const float la = min(-olala_softplus(raw.y), -1e-4f) * dt;
                const float trap = olala_sigmoid(raw.z);
                const size_t o = ((size_t)b * L + tl) * H + h;
                if (t < L) ((device float4*)sc)[o] = float4(la, trap * dt, (1.0f - trap) * dt, 0.0f);
                device float4* co = (device float4*)(cs + o * 2 * A);
                for (int i = 0; i < A / 4; ++i) {{
                    float4 th4;
                    for (int k = 0; k < 4; ++k) {{
                        const int a = 4 * i + k;
                        const float inc = tanh(ang[i][k]) * {math.pi}f * dt;
                        float th = carry[a] + simd_prefix_inclusive_sum(inc);
                        th = th - {2 * math.pi}f * rint(th * {0.5 / math.pi}f);
                        carry[a] = simd_shuffle(th, 31);
                        th4[k] = th;
                    }}
                    if (t < L) {{
                        co[i] = cos(th4);
                        co[A / 4 + i] = sin(th4);
                    }}
                }}
            }}
            for (int a = 0; a < A; ++a) if (lane == a) th_out[((size_t)b * H + h) * A + a] = carry[a];
        }} else {{
            const int row = blk - B * H;              // (b, t, j) with j < 2R
            const int j = row % (2 * R), bt = row / (2 * R);
            const float4 u = ((const device float4*)(proj + (size_t)bt * PW + NS + j * N))[lane];
            const float ss = simd_sum(dot(u, u));
            const float4 wn = ((const device float4*)(prm + (j < R ? O_BN : O_CN)))[lane];
            ((device float4*)(bc + (size_t)row * N))[lane] = u * rsqrt(ss / N + {1e-6}f) * wn;
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name="olala_m3c_prep",
            input_names=["proj", "prm", "st", "rng"],
            output_names=["sc", "cs", "th_out", "bc"],
            source=src,
            header=_HDR,
        )
    return _kernels[key](
        inputs=[proj, prm, st, rng],
        grid=((B * H + B * L * 2 * R + 7) // 8 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, H, 4), (B, L, H, 2, A), (B, H, A), (B, L, 2 * R, N)],
        output_dtypes=[mx.float32] * 4,
    )


def m3_chunked(proj, prm, st, H, P, N, R, A, n_static, rng=None):
    """Mamba-3 MIMO over a prompt in chunks of T tokens with simdgroup matrix products.

    Same interface and result as m3_prefill. Per chunk (rows (t, r) of the T*R queries / keys):
        O^T = exp(cum) * (S q^T)                           (inter-chunk, from the state)
            + V^T (M * (K Q^T))                            (intra-chunk, decay mask M)
        S   = exp(cum_T) S + (e * V)^T K                   (state update)
    where the trapezoid term of the next chunk's first token is folded into the last key's
    weight e. Threadgroup = (row, head, P / PS state rows), P / PS / 4 simdgroups; simdgroup g
    keeps state tiles (p block g / 2, n half g % 2) in registers for the whole prompt. The
    per-token inputs that do not depend on the state come from _m3c_prep. ~3x faster than
    m3_prefill at 2k tokens (limited by the simdgroup matrix throughput)."""
    B, L, _PW = proj.shape
    SW = P * N + R * N + P + A
    T, PS = M3C_T, M3C_PS
    TR, PL = T * R, P // PS
    NSG_ = PL // 4  # simdgroup g owns state tiles (p block g / 2, n half g % 2)
    NT = NSG_ * 32
    RB = 2 * TR // NSG_  # K / Q rows built per simdgroup
    assert N == 128 and A == 32 and TR == 16 and PL in (32, 64) and R % RB == 0
    if rng is None:
        rng = mx.broadcast_to(mx.array([0, L], dtype=mx.int32)[None], (B, 2))
    sc, cs, th, bc = _m3c_prep(proj, prm, st, rng, H, P, N, R, A, n_static)
    key = ("m3c", H, P, N, R, A, n_static, PS)
    if key not in _kernels:
        src = f"""
        constexpr int NT = {NT}, RB = {RB};
        constexpr int H = {H}, P = {P}, N = {N}, R = {R}, A = {A}, T = {T}, TR = {TR}, PL = {PL}, PS = {PS};
        constexpr int PH = 2 * P + 3, PW = {n_static} + 2 * R * N + A;
        constexpr int SW = {SW}, OK_ = P * N, OX = OK_ + R * N, OA = OX + P;
        constexpr int O_D = H, O_BN = 2 * H, O_CN = O_BN + N, O_BB = O_CN + N, O_CB = O_BB + H * R * N;
        constexpr int O_MX = O_CB + H * R * N, O_MZ = O_MX + H * R * P, O_MO = O_MZ + H * R * P;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        {_FRAG}
        const int blk = threadgroup_position_in_grid.x;
        const int b = blk / (H * PS), h = (blk / PS) % H, ph = blk % PS, p0 = ph * PL;
        const int L = proj_shape[1];
        const int s0 = rng[2 * b], s1 = rng[2 * b + 1];
        const device float* pr0 = proj + (size_t)b * L * PW;
        const device float4* sc0 = (const device float4*)sc + (size_t)b * L * H + h;
        const device float* cs0 = cs + ((size_t)b * L * H + h) * 2 * A;
        const device float* bc0 = bc + (size_t)b * L * 2 * R * N;
        const device float* sin_ = st + ((size_t)b * H + h) * SW;
        device float* sout = st_out + ((size_t)b * H + h) * SW;

        constexpr int NP = N + 4, PLP = PL + 4, TRP = TR + 4;   // padded rows (bank conflicts)
        threadgroup float Qs[TR][NP];
        threadgroup float Ks[TR][NP];
        threadgroup float Vs[TR][PLP];
        threadgroup float Ves[TR][PLP];
        threadgroup float Gs[TR][TRP];
        threadgroup float Os[PL][TRP];

        const float Dh = prm[O_D + h];
        const int pb = sg >> 1, nh = sg & 1;     // state tiles of this simdgroup
        // rows built by this simdgroup: token tb, K or Q, r in [r0, r0 + RB)
        const int tb = sg / (2 * R / RB), isq = (sg / (R / RB)) & 1, r0 = (sg % (R / RB)) * RB;
        float4 bb[RB];
        for (int r = 0; r < RB; ++r) bb[r] = ((const device float4*)(prm + (isq ? O_CB : O_BB) + (h * R + r0 + r) * N))[lane];
        auto load_sc = [&](int t) {{ return t >= s0 && t < s1 ? sc0[(size_t)t * H] : float4(0); }};

        simdgroup_float8x8 Sm[8];
        for (int j = 0; j < 8; ++j) simdgroup_load(Sm[j], sin_ + (p0 + pb * 8) * N + (nh * 8 + j) * 8, N);

        // trapezoid term of the first token: S += delta_s0 * v_prev^T k_prev (rows 0..R-1, padded to 8)
        {{
            const float d0 = load_sc(s0).z;
            for (int i = tid; i < 8 * N; i += NT) {{
                const int r = i / N, n = i % N;
                Ks[r][n] = r < R ? sin_[OK_ + r * N + n] : 0.0f;
            }}
            for (int i = tid; i < 8 * PL; i += NT) {{
                const int r = i / PL, pl = i % PL;
                Vs[r][pl] = r < R ? d0 * sin_[OX + p0 + pl] * prm[O_MX + (h * R + r) * P + p0 + pl] : 0.0f;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 vt, kt;
            simdgroup_load(vt, &Vs[0][pb * 8], PLP, 0, true);
            for (int j = 0; j < 8; ++j) {{
                simdgroup_load(kt, &Ks[0][(nh * 8 + j) * 8], NP);
                simdgroup_multiply_accumulate(Sm[j], vt, kt, Sm[j]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // per-thread constants for the output (fixed (t, p)) and for the V rows
        const int yt = tid / PL, ypl = tid % PL, yp = p0 + ypl;
        float cmx[R], cmz[R], cmo[R], vmx[2];
        for (int r = 0; r < R; ++r) {{
            cmx[r] = prm[O_MX + (h * R + r) * P + yp];
            cmz[r] = prm[O_MZ + (h * R + r) * P + yp];
            cmo[r] = prm[O_MO + (h * R + r) * P + yp];
        }}
        for (int k = 0; k < 2; ++k) {{
            const int row = (tid + NT * k) / PL;
            vmx[k] = prm[O_MX + (h * R + row % R) * P + p0 + ypl];
        }}
        const bool rot = (lane & 15) < 8;
        const int ai = 4 * (lane & 7);

        // inputs of chunk c (fetched one chunk ahead)
        float4 scv_n, cs_n, sn_n, bc_n[RB];
        float x_n0, x_n1;
#define M3C_FETCH(c) {{ \
            scv_n = load_sc(lane <= T ? (c) + lane : -1); \
            const int tc_ = min((c) + tb, L - 1); \
            cs_n = *(const device float4*)(cs0 + (size_t)tc_ * H * 2 * A + ai); \
            sn_n = *(const device float4*)(cs0 + (size_t)tc_ * H * 2 * A + A + ai); \
            for (int r = 0; r < RB; ++r) bc_n[r] = ((const device float4*)(bc0 + ((size_t)tc_ * 2 * R + isq * R + r0 + r) * N))[lane]; \
            x_n0 = pr0[(size_t)min((c) + xs0, L - 1) * PW + h * PH + P + p0 + ypl]; \
            x_n1 = pr0[(size_t)min((c) + xs1, L - 1) * PW + h * PH + P + p0 + ypl]; \
            const int ty_ = min((c) + min(yt, T - 1), L - 1); \
            (void)ty_; }}
        const int xs0 = tid / PL / R, xs1 = (tid + NT) / PL / R;
        if (s1 > s0) M3C_FETCH(s0);
        for (int c0 = s0; c0 < s1; c0 += T) {{
            const float4 scv = scv_n;
            const int ty = min(c0 + min(yt, T - 1), L - 1);
            const float yx = pr0[(size_t)ty * PW + h * PH + P + yp], yz = pr0[(size_t)ty * PW + h * PH + yp];
            // token scalars: lane t < T for token c0 + t, lane T for the next chunk's first token
            const float la = scv.x, gam = scv.y, del = scv.z;
            const float cum = simd_prefix_inclusive_sum(lane < T ? la : 0.0f);
            const float cumT = simd_shuffle(cum, T - 1);
            const float gw = gam + simd_shuffle_down(del, 1);       // weight of u_t for later tokens
            const float e = lane < T ? exp(cumT - cum) * gw : 0.0f;  // weight of u_t in the chunk's state update

            // build: K / Q rows (bias, rotate) of token tb, V and e * V
            {{
                const int t = c0 + tb;
                const bool ok = t < s1;
                for (int rr = 0; rr < RB; ++rr) {{
                    const int r = r0 + rr;
                    float4 val = ok ? bc_n[rr] + bb[rr] : 0.0f;
                    const float4 partner = simd_shuffle_xor(val, 16);
                    if (rot) val = (lane < 16) ? (val * cs_n - partner * sn_n) : (partner * sn_n + val * cs_n);
                    threadgroup float4* dst = (threadgroup float4*)(isq ? &Qs[tb * R + r][0] : &Ks[tb * R + r][0]);
                    dst[lane] = val;
                    if (t == s1 - 1 && !isq && ph == 0) ((device float4*)(sout + OK_ + r * N))[lane] = val;
                }}
                for (int k = 0; k < 2; ++k) {{
                    const int i = tid + NT * k;
                    const int row = i / PL, pl = i % PL, s = row / R;
                    const int ts = c0 + s;
                    const float x = ts < s1 ? (k == 0 ? x_n0 : x_n1) : 0.0f;
                    const float v = x * vmx[k];
                    Vs[row][pl] = v;
                    Ves[row][pl] = v * simd_shuffle(e, s);
                    if (row % R == 0 && ts == s1 - 1) sout[OX + p0 + pl] = x;
                }}
            }}
            if (c0 + T < s1) M3C_FETCH(c0 + T);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // inter-chunk part from the state, then the state update (same registers)
            simdgroup_float8x8 oi[2], tq, tk;
            for (int i = 0; i < 2; ++i) {{
                oi[i] = simdgroup_float8x8(0);
                for (int j = 0; j < 8; ++j) {{
                    simdgroup_load(tq, &Qs[i * 8][(nh * 8 + j) * 8], NP, 0, true);
                    simdgroup_multiply_accumulate(oi[i], Sm[j], tq, oi[i]);
                }}
                // column (t, r) scaled by exp(cum_t)
                auto el = oi[i].thread_elements();
                const int tc = (i * 8 + _FN) / R;   // _FN and _FN + 1 belong to the same token (R == 4)
                const float sc_ = exp(simd_shuffle(cum, tc));
                el[0] *= sc_; el[1] *= sc_;
                oi[i].thread_elements() = el;
            }}
            {{
                const float ac = exp(cumT);
                simdgroup_float8x8 vt[2];
                for (int k = 0; k < 2; ++k) simdgroup_load(vt[k], &Ves[k * 8][pb * 8], PLP, 0, true);
                for (int j = 0; j < 8; ++j) {{
                    Sm[j].thread_elements()[0] *= ac;
                    Sm[j].thread_elements()[1] *= ac;
                    for (int k = 0; k < 2; ++k) {{
                        simdgroup_load(tk, &Ks[k * 8][(nh * 8 + j) * 8], NP);
                        simdgroup_multiply_accumulate(Sm[j], vt[k], tk, Sm[j]);
                    }}
                }}
            }}
            // intra-chunk scores G^T = K Q^T with the decay mask (simdgroups 0..3, one tile each)
            if (sg < 4) {{
                const int ki = sg >> 1, qi = sg & 1;
                simdgroup_float8x8 g = simdgroup_float8x8(0);
                for (int j = 0; j < N / 8; ++j) {{
                    simdgroup_load(tk, &Ks[ki * 8][j * 8], NP);
                    simdgroup_load(tq, &Qs[qi * 8][j * 8], NP, 0, true);
                    simdgroup_multiply_accumulate(g, tk, tq, g);
                }}
                auto el = g.thread_elements();
                const int s = (ki * 8 + _FM) / R, t = (qi * 8 + _FN) / R;
                const float cs_ = simd_shuffle(cum, s), ct = simd_shuffle(cum, t);
                const float gs = simd_shuffle(gw, s), gt = simd_shuffle(gam, t);
                const float m = s < t ? exp(ct - cs_) * gs : (s == t ? gt : 0.0f);
                el[0] *= m; el[1] *= m;
                g.thread_elements() = el;
                simdgroup_store(g, &Gs[ki * 8][qi * 8], TRP);
            }}
            if (nh == 1) for (int i = 0; i < 2; ++i) simdgroup_store(oi[i], &Os[pb * 8][i * 8], TRP);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (nh == 0) {{
                simdgroup_float8x8 vt, gg, o2;
                for (int i = 0; i < 2; ++i) {{
                    simdgroup_load(o2, &Os[pb * 8][i * 8], TRP);
                    auto a = oi[i].thread_elements();
                    const auto c = o2.thread_elements();
                    a[0] += c[0]; a[1] += c[1];
                    oi[i].thread_elements() = a;
                    for (int k = 0; k < 2; ++k) {{
                        simdgroup_load(vt, &Vs[k * 8][pb * 8], PLP, 0, true);
                        simdgroup_load(gg, &Gs[k * 8][i * 8], TRP);
                        simdgroup_multiply_accumulate(oi[i], vt, gg, oi[i]);
                    }}
                    simdgroup_store(oi[i], &Os[pb * 8][i * 8], TRP);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // y_t = sum_r w_tr * (D v_tr + O_tr)
            if (yt < T && c0 + yt < s1) {{
                const float x = yx, z = yz;
                float acc = 0;
                for (int r = 0; r < R; ++r) {{
                    const float zz = z * cmz[r];
                    acc += zz / (1.0f + exp(-zz)) * cmo[r] * (Dh * x * cmx[r] + Os[ypl][yt * R + r]);
                }}
                y[((size_t)b * L + c0 + yt) * H * P + h * P + yp] = acc;
            }}
        }}
#undef M3C_FETCH
        for (int j = 0; j < 8; ++j) simdgroup_store(Sm[j], sout + (p0 + pb * 8) * N + (nh * 8 + j) * 8, N);
        if (ph == 0 && sg == 0) sout[OA + lane] = th[((size_t)b * H + h) * A + lane];
        if (s1 <= s0) {{
            for (int i = tid; i < PL; i += NT) sout[OX + p0 + i] = sin_[OX + p0 + i];
            if (ph == 0) for (int i = tid; i < R * N; i += NT) sout[OK_ + i] = sin_[OK_ + i];
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name="olala_m3c",
            input_names=["proj", "prm", "st", "rng", "sc", "cs", "th", "bc"],
            output_names=["y", "st_out"],
            source=src,
            header=_HDR,
        )
    y, st_out = _kernels[key](
        inputs=[proj, prm, st, rng, sc, cs, th, bc],
        grid=(B * H * PS * NSG_ * 32, 1, 1),
        threadgroup=(NSG_ * 32, 1, 1),
        output_shapes=[(B, L, H * P), (B, H, SW)],
        output_dtypes=[mx.float32, mx.float32],
        init_value=0,
    )
    return y, st_out


# --------------------------------------------------------------------------
# MoE experts
# --------------------------------------------------------------------------


def _topk_code(top_k, scaling):
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
_MOE_ATOMIC = int(
    os.environ.get(
        "OLALA_MOE_ATOMIC", "0" if os.environ.get("OLALA_DETERMINISTIC") == "1" else "1"
    )
)


MOE_FB = 64  # expert hidden units per threadgroup in moe_fused


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
    LPRU = max(
        1, min(32, (NT * (D // chu)) // (FB * max(1, D // chu)) if False else NT // FB)
    )
    while LPRU > 1 and (D // chu) % LPRU:
        LPRU //= 2
    key = ("moe_fused2", E, F, D, wu.key(), wd.key(), top_k, FB, NSG, x0_stride)
    if key not in _kernels:

        def xu(t, j):
            return f"*(const threadgroup float4*)(xs + k + 4 * ({j}))"

        def xd(t, j):
            return f"*(const threadgroup float4*)(hs + (k - f0) + 4 * ({j}))"

        dotu = _dot_code(wu.fmt, "wu", D, wu.gs or 0, xu, 1, "erow", "k").replace(
            "acc[r][t]", "acc"
        )
        dotd = _dot_code(wd.fmt, "wd", F, wd.gs or 0, xd, 1, "erow", "k").replace(
            "acc[r][t]", "acc"
        )
        src = f"""
        constexpr int E = {E}, F = {F}, D = {D}, TOPK = {top_k}, FB = {FB}, NB = {
            NB
        }, NT = {NT}, LPRU = {LPRU};
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
            {
            "atomic_fetch_add_explicit(&parts[(size_t)t * D + o], acc, memory_order_relaxed);"
            if _MOE_ATOMIC
            else "parts[((size_t)t * TOPK * NB + kk * NB + fb) * D + o] = acc;"
        }
        }}
        """
        names = (
            ["logits", "ebias", "x0", "wu"]
            + (["wu_s", "wu_b"] if wu.fmt.startswith("q") else [])
            + ["wd"]
            + (["wd_s", "wd_b"] if wd.fmt.startswith("q") else [])
        )
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_moe_fused2_{abs(hash(key)) % 10**12}a{_MOE_ATOMIC}",
            input_names=names,
            output_names=["parts"],
            source=src,
            header=_HDR,
            atomic_outputs=bool(_MOE_ATOMIC),
        )
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
        grid=(T * top_k * NB * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(T, top_k * NB, D)],
        output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# attention (V layers)
# --------------------------------------------------------------------------


def v_prep(
    big,
    small,
    kprev,
    vprev,
    prm,
    pos,
    Hq,
    Hk,
    Dh,
    Rk,
    n_noise,
    window,
    T,
    kv_dtype,
    kbuf=None,
    vbuf=None,
    wpos=None,
):
    """Returns q (T,Hq,Dh) f32 (normed, scalable-softmax scaled), k / v (T,Hk,1,Dh) kv_dtype,
    raw k / v (T,1,Hk,Dh) f32 for the next token shift.
    With kbuf / vbuf (full cache buffers (B,Hk,cap,Dh)) the new k / v are written in place at
    position wpos[t] instead (see model.KVCache) and k / v are not returned."""
    inplace = kbuf is not None
    key = ("v_prep", Hq, Hk, Dh, Rk, str(kv_dtype), inplace)
    if key not in _kernels:
        ot = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[kv_dtype]
        nbig = Hq * Dh + 2 * Rk * Dh + (Hq - Hk) * Dh
        nsmall = 2 * Hk * Rk + 2 * Hk + n_noise
        if inplace:
            kdst = f"((device {ot}*)kbuf)[(((size_t)t * Hk + h) * kbuf_shape[2] + wpos[t]) * Dh + tid]"
            vdst = f"((device {ot}*)vbuf)[(((size_t)t * Hk + h) * vbuf_shape[2] + wpos[t]) * Dh + tid]"
        else:
            kdst = "k_out[((size_t)t * Hk + h) * Dh + tid]"
            vdst = "v_out[((size_t)t * Hk + h) * Dh + tid]"
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
            {vdst} = ({ot})(av * vp + (1 - av) * vr);
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
            {kdst} = ({ot})(val * rn * kn[tid]);
        }}
        """
        inputs = ["big", "small", "kprev", "vprev", "qn", "kn", "scaler", "pos"]
        outputs = (
            ["q_out", "kraw_out", "vraw_out"]
            if inplace
            else ["q_out", "k_out", "v_out", "kraw_out", "vraw_out"]
        )
        if inplace:
            inputs += ["kbuf", "vbuf", "wpos"]
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_v_prep_{abs(hash(key)) % 10**12}",
            input_names=inputs,
            output_names=outputs,
            source=src,
            header=_HDR,
        )
    ins = [big, small, kprev, vprev, prm["q_norm"], prm["k_norm"], prm["scaler"], pos]
    if inplace:
        return _kernels[key](
            inputs=ins + [kbuf, vbuf, wpos],
            grid=(T * (Hq + Hk) * Dh, 1, 1),
            threadgroup=(Dh, 1, 1),
            output_shapes=[(T, Hq, Dh), (T, 1, Hk, Dh), (T, 1, Hk, Dh)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    return _kernels[key](
        inputs=ins,
        grid=(T * (Hq + Hk) * Dh, 1, 1),
        threadgroup=(Dh, 1, 1),
        output_shapes=[
            (T, Hq, Dh),
            (T, Hk, 1, Dh),
            (T, Hk, 1, Dh),
            (T, 1, Hk, Dh),
            (T, 1, Hk, Dh),
        ],
        output_dtypes=[mx.float32, kv_dtype, kv_dtype, mx.float32, mx.float32],
    )


ATT_NW = 4  # simdgroups per attention threadgroup


ATT_BLK = 256  # keys per attention threadgroup


def _att_blk(S, Hk, T):
    """Keys per threadgroup (smaller blocks for more parallelism measured slower overall)."""
    return ATT_BLK


def attn_decode(
    q, keys, values, kstart, scale, softcap, T, Hq, Hk, Dh, S=None, slen=None
):
    """q (T,Hq,Dh) f32; keys / values (B,Hk,S,Dh) (possibly strided views).
    Valid keys of row t: [kstart[t], S). Returns partial o (T,Hq,NS,Dh), m, l (T,Hq,NS).

    One threadgroup per (row, kv head, block of keys); each simdgroup walks keys of the block,
    lane = 4 consecutive dims, so a key / value row is one coalesced load; the G query heads
    sharing the kv head are processed together."""
    S = S or keys.shape[2]
    if slen is None:
        slen = mx.array([S], dtype=mx.int32)
    BLK = _att_blk(S, Hk, T)
    NS = (S + BLK - 1) // BLK
    G = Hq // Hk
    assert Dh == 128
    key = ("attn3", Hq, Hk, Dh, str(keys.dtype), BLK)
    if key not in _kernels:
        vt = {"bfloat16": "bfloat4", "float16": "half4", "float32": "float4"}[
            str(keys.dtype).split(".")[-1]
        ]
        src = f"""
        constexpr int Hq = {Hq}, Hk = {Hk}, Dh = {Dh}, G = {G}, BLK = {BLK}, NW = {ATT_NW};
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int S = slen[0];
        const int NS = (S + BLK - 1) / BLK;
        const int t = blk / (Hk * NS);
        const int hk = (blk / NS) % Hk;
        const int sp = blk % NS;
        const int j0 = max(sp * BLK, kstart[t]);
        const int j1 = min((sp + 1) * BLK, S);
        threadgroup float4 qs[G][Dh / 4];
        threadgroup float mo[NW][G];
        threadgroup float lo[NW][G];
        threadgroup float4 oo[NW][G][Dh / 4];
        for (int i = tid; i < G * Dh / 4; i += NW * 32)
            qs[i / (Dh / 4)][i % (Dh / 4)] = ((const device float4*)(q + ((size_t)t * Hq + hk * G) * Dh))[i] * {scale}f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device {vt}* kb = (const device {vt}*)(keys + (size_t)t * keys_strides[0] + (size_t)hk * keys_strides[1]);
        const device {vt}* vb = (const device {vt}*)(values + (size_t)t * values_strides[0] + (size_t)hk * values_strides[1]);
        const size_t ks4 = keys_strides[2] / 4, vs4 = values_strides[2] / 4;
        float m[G], l[G];
        float4 o[G];
        for (int g = 0; g < G; ++g) {{ m[g] = -INFINITY; l[g] = 0; o[g] = 0; }}
        // simdgroup sg takes chunks of 32 keys: scores with lane = key, values with lane = 4 dims
        for (int jb = j0 + sg * 32; jb < j1; jb += NW * 32) {{
            const int j = jb + lane;
            float sc[G];
            if (j < j1) {{
                const device {vt}* kr = kb + (size_t)j * ks4;
                for (int g = 0; g < G; ++g) sc[g] = 0;
                for (int d4 = 0; d4 < Dh / 4; ++d4) {{
                    const float4 kv = float4(kr[d4]);
                    for (int g = 0; g < G; ++g) sc[g] += dot(kv, qs[g][d4]);
                }}
                for (int g = 0; g < G; ++g) sc[g] = tanh(sc[g] / {softcap}f) * {softcap}f;
            }} else {{
                for (int g = 0; g < G; ++g) sc[g] = -INFINITY;
            }}
            float pj[G];
            for (int g = 0; g < G; ++g) {{
                const float mn = max(m[g], simd_max(sc[g]));
                const float corr = exp(m[g] - mn);
                pj[g] = exp(sc[g] - mn);
                l[g] = l[g] * corr + simd_sum(pj[g]);
                o[g] *= corr;
                m[g] = mn;
            }}
            const int nk = min(32, j1 - jb);
            for (int u = 0; u < nk; ++u) {{
                const float4 vv = float4(vb[(size_t)(jb + u) * vs4 + lane]);
                for (int g = 0; g < G; ++g) o[g] += simd_shuffle(pj[g], u) * vv;
            }}
        }}
        for (int g = 0; g < G; ++g) {{
            if (lane == 0) {{ mo[sg][g] = m[g]; lo[sg][g] = l[g]; }}
            oo[sg][g][lane] = o[g];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < G) {{
            const int g = sg;
            float M = -INFINITY;
            for (int w = 0; w < NW; ++w) M = max(M, mo[w][g]);
            float L = 0; float4 O = 0;
            for (int w = 0; w < NW; ++w) {{
                const float c = (mo[w][g] == -INFINITY) ? 0.0f : exp(mo[w][g] - M);
                L += lo[w][g] * c; O += oo[w][g][lane] * c;
            }}
            const size_t oi = ((size_t)t * Hq + hk * G + g) * NS + sp;
            ((device float4*)(o_out + oi * Dh))[lane] = O;
            if (lane == 0) {{ m_out[oi] = M; l_out[oi] = L; }}
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_attn2_{abs(hash(key)) % 10**12}",
            input_names=["q", "keys", "values", "kstart", "slen"],
            output_names=["o_out", "m_out", "l_out"],
            source=src,
            header=_HDR,
            ensure_row_contiguous=False,
        )
    return _kernels[key](
        inputs=[q, keys, values, kstart, slen],
        grid=(T * Hk * NS * ATT_NW * 32, 1, 1),
        threadgroup=(ATT_NW * 32, 1, 1),
        output_shapes=[(T, Hq, NS, Dh), (T, Hq, NS), (T, Hq, NS)],
        output_dtypes=[mx.float32] * 3,
    )


FA_BQ = 32  # queries per threadgroup in attn_prefill
FA_BK = 32  # keys per block


def attn_prefill(q, k, v, scale, softcap, window):
    """Causal soft-capped attention for a prompt chunk, flash-attention style.

    q (B, Hq, L, D) (any float dtype), k / v (B, Hk, T, D) with the chunk's queries at
    positions T - L .. T - 1. Keys within ``window`` of the query. Returns (B, Hq, L, D) in
    v's dtype. One threadgroup per (row, kv head, block of FA_BQ queries) processes the G
    query heads sharing the kv head: thread = (head, query, quarter of the head dim)."""
    B, Hq, L, D = q.shape
    Hk = k.shape[1]
    G = Hq // Hk
    assert D == 128 and G * FA_BQ * 4 <= 1024
    key = ("fa", Hq, Hk, D, str(q.dtype), str(k.dtype), str(v.dtype), int(window))
    if key not in _kernels:
        kt = {"bfloat16": "bfloat", "float16": "half", "float32": "float"}[
            str(k.dtype).split(".")[-1]
        ]
        ot = {"bfloat16": "bfloat", "float16": "half", "float32": "float"}[
            str(v.dtype).split(".")[-1]
        ]
        src = f"""
        constexpr int Hq = {Hq}, Hk = {Hk}, D = {D}, G = {G}, BQ = {FA_BQ}, BK = {FA_BK}, DP = D / 4;
        constexpr int WIN = {int(window)};
        const uint tid = thread_position_in_threadgroup.x;
        const int g = tid / (BQ * 4), qr = (tid / 4) % BQ, part = tid % 4;
        const uint blk = threadgroup_position_in_grid.x;
        const int L = q_shape[2], T = k_shape[2];
        const int nqb = (L + BQ - 1) / BQ;
        const int b = blk / (Hk * nqb);
        const int hk = (blk / nqb) % Hk;
        const int qb = blk % nqb;
        const int hq = hk * G + g;
        const int qi = qb * BQ + qr;                 // query index in the chunk
        const int qpos = T - L + min(qi, L - 1);     // its key position
        threadgroup {kt} ks[BK][D];
        threadgroup {kt} vs[BK][D];
        float qv[DP], o[DP];
        const size_t qoff = (((size_t)b * Hq + hq) * L + min(qi, L - 1)) * D + part * DP;
        for (int d = 0; d < DP; ++d) {{ qv[d] = float(q[qoff + d]) * {scale}f; o[d] = 0; }}
        float m = -INFINITY, l = 0;
        // key range of the whole threadgroup
        const int qpos_lo = T - L + qb * BQ, qpos_hi = T - L + min(qb * BQ + BQ, L) - 1;
        const int j_lo = max(0, qpos_lo - WIN), j_hi = qpos_hi;
        const size_t kbase = (size_t)b * k_strides[0] + (size_t)hk * k_strides[1];
        const size_t vbase = (size_t)b * v_strides[0] + (size_t)hk * v_strides[1];
        for (int j0 = (j_lo / BK) * BK; j0 <= j_hi; j0 += BK) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < BK * D; i += G * BQ * 4) {{
                const int jj = j0 + i / D;
                ks[i / D][i % D] = jj < T ? k[kbase + (size_t)jj * k_strides[2] + i % D] : {kt}(0);
                vs[i / D][i % D] = jj < T ? v[vbase + (size_t)jj * v_strides[2] + i % D] : {kt}(0);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sc[BK];
            float bm = -INFINITY;
            for (int j = 0; j < BK; ++j) {{
                float sp = 0;
                for (int d = 0; d < DP; ++d) sp += qv[d] * float(ks[j][part * DP + d]);
                sp += simd_shuffle_xor(sp, 1);
                sp += simd_shuffle_xor(sp, 2);
                const int jp = j0 + j;
                const bool ok = jp <= qpos && jp >= qpos - WIN && jp < T;
                sc[j] = ok ? tanh(sp / {softcap}f) * {softcap}f : -INFINITY;
                bm = max(bm, sc[j]);
            }}
            const float mn = max(m, bm);
            if (mn == -INFINITY) continue;
            const float corr = exp(m - mn);
            l *= corr;
            for (int d = 0; d < DP; ++d) o[d] *= corr;
            for (int j = 0; j < BK; ++j) {{
                const float pj = exp(sc[j] - mn);
                l += pj;
                for (int d = 0; d < DP; ++d) o[d] += pj * float(vs[j][part * DP + d]);
            }}
            m = mn;
        }}
        if (qi < L) {{
            const float inv = 1.0f / l;
            const size_t ooff = (((size_t)b * Hq + hq) * L + qi) * D + part * DP;
            for (int d = 0; d < DP; ++d) out[ooff + d] = ({ot})(o[d] * inv);
        }}
        """
        _kernels[key] = mx.fast.metal_kernel(
            name=f"olala_fa_{abs(hash(key)) % 10**12}",
            input_names=["q", "k", "v"],
            output_names=["out"],
            source=src,
            header=_HDR,
            ensure_row_contiguous=False,
        )
    q = mx.contiguous(q)
    nqb = (L + FA_BQ - 1) // FA_BQ
    return _kernels[key](
        inputs=[q, k, v],
        grid=(B * Hk * nqb * G * FA_BQ * 4, 1, 1),
        threadgroup=(G * FA_BQ * 4, 1, 1),
        output_shapes=[(B, Hq, L, D)],
        output_dtypes=[v.dtype],
    )[0]


def attn_combine(
    o, m, l, small, big, Hq, Hk, Dh, snr, n_noise, gate_bias, T, small_off, big_off
):
    """Merge the key splits, apply the differential combination and the output gate.
    Returns (T, n_sig * Dh) f32."""
    NS = o.shape[2]
    n_sig = Hq - n_noise
    key = (
        "attn_comb",
        Hq,
        Dh,
        snr,
        n_noise,
        NS,
        small_off,
        big_off,
        small.shape[1],
        big.shape[1],
    )
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
            input_names=["o", "m", "l", "small", "big"],
            output_names=["out"],
            source=src,
            header=_HDR,
        )
    return _kernels[key](
        inputs=[o, m, l, small, big],
        grid=(T * n_sig * Dh, 1, 1),
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
                    "q_norm": (1.0 + mx_.q_norm.norm.weight.astype(f32))
                    if zc
                    else mx_.q_norm.norm.weight.astype(f32),
                    "k_norm": (1.0 + mx_.k_norm.norm.weight.astype(f32))
                    if zc
                    else mx_.k_norm.norm.weight.astype(f32),
                    "scaler": mx_.softmax_scaler.astype(f32),
                }
                L["attn"] = mx_
            L["mixer_proj"] = W(layer.mixer_proj)
            L["geo_mixer"] = geo_params(layer.geodesic_mixer, layer.layer_idx)
            L["geo_mlp"] = geo_params(layer.geodesic_mlp, layer.layer_idx)
            moe = layer.mlp
            L["router"] = W(moe.moe_gate)
            L["in_cat"] = W(moe.in_cat)
            L["out_cat"] = W(moe.out_cat)
            L["up"] = W(moe.experts.experts)
            L["down"] = W(moe.experts.output_experts)
            L["ebias"] = moe.expert_bias.astype(f32)
            L["moe"] = moe
            self.layers.append(L)
        # materialize everything now: lazily created arrays belong to the creating thread's
        # stream, and decoding may run on another thread (mlx-engine's generation thread)
        mx.eval(
            [
                [v for v in L["prm"].values()]
                + [L["geo_mixer"], L["geo_mlp"], L["ebias"]]
                for L in self.layers
            ]
        )
        mx.eval(
            [
                w.arrays()
                for L in self.layers
                for w in (
                    L[k]
                    for k in (
                        "in",
                        "in_small",
                        "mixer_proj",
                        "router",
                        "in_cat",
                        "out_cat",
                        "up",
                        "down",
                    )
                    if k in L
                )
            ]
        )


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
    B, L, _D = x.shape
    if (
        L != 1
        or B > MAX_ROWS
        or ssm_mask is not None
        or cache is None
        or cache[0] is None
    ):
        return False
    for i in model.attn_idx:
        c = cache[i]
        if c is None or not hasattr(c, "caches"):
            return False
        c = c.caches[0]
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


# fused paths that can be switched off (debugging / comparison)
_INPLACE_KV = os.environ.get("OLALA_INPLACE_KV", "1") == "1"
# geodesic sums accumulated (atomically) by the GEMV producing the update, so the next GEMV's
# prologue needs no reduction; off in deterministic mode (summation order varies)
_GEO_STATS = (
    os.environ.get("OLALA_GEO_STATS", "1") == "1"
    and os.environ.get("OLALA_DETERMINISTIC") != "1"
)


def _moe_block(L, resid, gmix, T, D, deps=(), gst=None):
    """geodesic(resid, gmix) -> MoE. Returns (resid', g, gst') with g the MoE output and gst'
    its geodesic sums for the next layer (None when not available)."""
    li = L["li"]
    moe = L["moe"]
    if gst is not None:
        logits, hcat, resid = gemv(
            [resid, gmix, L["geo_mixer"], gst],
            [L["router"], L["in_cat"]],
            prologue="geos",
            LI=li,
            T=T,
            K=D,
            deps=deps,
        )
    else:
        logits, hcat, resid = gemv(
            [resid, gmix, L["geo_mixer"]],
            [L["router"], L["in_cat"]],
            prologue="geo",
            LI=li,
            T=T,
            K=D,
            deps=deps,
        )
    sx = resid if _GEO_STATS else None
    K_out = moe.d_latent + moe.d_shared
    if T > _gemv_limit([L["up"], L["down"]]):
        # many rows: sorted gather (q)matmul over the selected experts
        from . import kernels

        inds, probs = kernels.moe_route(logits, L["ebias"], moe.top_k, moe.scaling)
        x0 = hcat[:, : moe.d_latent]
        if moe.experts.experts.weight.dtype in (mx.bfloat16, mx.float16):
            x0 = x0.astype(moe.experts.experts.weight.dtype)
        e = kernels.moe_experts(x0, inds, probs.astype(x0.dtype), moe.experts).astype(
            mx.float32
        )
        g = gemv(
            [e, hcat],
            [L["out_cat"]],
            prologue="relu2cat",
            Ka=moe.d_latent,
            sa=moe.d_latent,
            sb=hcat.shape[1],
            T=T,
            K=K_out,
        )[0]
        return resid, g, None
    parts = moe_fused(
        logits,
        L["ebias"],
        hcat,
        hcat.shape[1],
        L["up"],
        L["down"],
        moe.top_k,
        moe.scaling,
        T,
    )
    if _MOE_ATOMIC:  # parts is the summed expert output (T, d_latent)
        outs = gemv(
            [parts, hcat],
            [L["out_cat"]],
            prologue="relu2cat",
            Ka=moe.d_latent,
            sa=moe.d_latent,
            sb=hcat.shape[1],
            T=T,
            K=K_out,
            stats_x=sx,
        )
    else:  # per-threadgroup partials, summed in a fixed order by the GEMV prologue
        outs = gemv(
            [parts, hcat],
            [L["out_cat"]],
            prologue="moecat",
            Ka=moe.d_latent,
            sa=parts.shape[1],
            sb=hcat.shape[1],
            T=T,
            K=K_out,
        )
        sx = None
    return resid, outs[0], (outs[-1] if sx is not None else None)


def _in_gemv(segs, resid, g, prev_geo, prev_li, gst, T, D):
    if g is None:
        return gemv([resid], segs, prologue="norm", T=T, K=D)
    if gst is not None:
        return gemv(
            [resid, g, prev_geo, gst], segs, prologue="geos", LI=prev_li, T=T, K=D
        )
    return gemv([resid, g, prev_geo], segs, prologue="geo", LI=prev_li, T=T, K=D)


def _m_layer(L, resid, g, prev_geo, prev_li, st, T, D, gst=None):
    """One M layer. Returns (resid', g', state', gst')."""
    H, P, N, R, A, n_static = L["dims"]
    outs = _in_gemv([L["in"]], resid, g, prev_geo, prev_li, gst, T, D)
    resid = outs[-1]
    if _GEO_STATS and not (T == 1 and L["mixer_proj"].scales is None):
        # separate mixer_proj GEMV (as fast as the fused one for quantized weights; the fused
        # step wins for bf16 at one row) so that it produces the sums
        y, st = m3_step(outs[0], L["prm"]["m3"], st, H, P, N, R, A, n_static)
        gmix, gmst = gemv([y], [L["mixer_proj"]], T=T, stats_x=resid)
        resid, g, gst = _moe_block(L, resid, gmix, T, D, gst=gmst)
        return resid, g, st, gst
    if T == 1:
        # mixer_proj fused into the step: the threadgroup of each head adds its contribution
        # atomically into gmix; the next GEMV depends on the step's state output for ordering
        gmix = zeros_f32((T, D))
        st = m3_step(
            outs[0],
            L["prm"]["m3"],
            st,
            H,
            P,
            N,
            R,
            A,
            n_static,
            wmp=L["mixer_proj"],
            gmix=gmix,
        )
        resid, g, gst = _moe_block(L, resid, gmix, T, D, deps=(st,))
    else:
        y, st = m3_step(outs[0], L["prm"]["m3"], st, H, P, N, R, A, n_static)
        gmix = gemv([y], [L["mixer_proj"]], T=T)[0]
        resid, g, gst = _moe_block(L, resid, gmix, T, D)
    return resid, g, st, gst


def _v_layer(L, resid, g, prev_geo, prev_li, cache, pos, T, D, shared, gst=None):
    """One V layer. ``shared`` carries per-token values common to all attention layers."""
    segs = [L["in"], L["in_small"]]
    outs = _in_gemv(segs, resid, g, prev_geo, prev_li, gst, T, D)
    resid = outs[-1]
    big, small = outs[0], outs[1]
    att = L["attn"]
    Hq, Hk, Dh, Rk = att.n_heads, att.n_kv, att.head_dim, att.rank
    kc, sc = cache.caches[0], cache.caches[1]
    kv_dtype = kc.keys.dtype
    if sc[0] is None:
        kprev = vprev = mx.zeros((T, 1, Hk, Dh), mx.float32)
    else:
        kprev, vprev = sc[0].astype(mx.float32), sc[1].astype(mx.float32)
    args = (
        big,
        small,
        kprev,
        vprev,
        L["prm"],
        pos,
        Hq,
        Hk,
        Dh,
        Rk,
        att.n_noise,
        att.window,
        T,
        kv_dtype,
    )
    if _INPLACE_KV and hasattr(kc, "can_append_inplace") and kc.can_append_inplace(1):
        if shared.get("wpos_off") != kc.offset:
            shared["wpos_off"] = kc.offset
            shared["wpos"] = mx.full((T,), kc.offset, dtype=mx.int32)
        q, kraw, vraw = v_prep(*args, kbuf=kc.keys, vbuf=kc.values, wpos=shared["wpos"])
        kc.commit_inplace(1)
        keys, values, S_len = kc.keys, kc.values, kc.offset
    else:
        q, k_new, v_new, kraw, vraw = v_prep(*args)
        keys, values = kc.update_and_fetch(k_new, v_new)
        S_len = keys.shape[2]
    sc[0], sc[1] = kraw, vraw
    if "kstart" not in shared:
        lp = getattr(kc, "left_padding", None)
        ks = max(S_len - 1 - att.window, 0)
        shared["kstart"] = (
            mx.maximum(lp, ks) if lp is not None else mx.full((T,), ks)
        ).astype(mx.int32)
        shared["slen"] = mx.array([S_len], dtype=mx.int32)
    o, m, l = attn_decode(
        q,
        keys,
        values,
        shared["kstart"],
        att.scale,
        att.softcap,
        T,
        Hq,
        Hk,
        Dh,
        S=S_len,
        slen=shared["slen"],
    )
    ymix = attn_combine(
        o,
        m,
        l,
        small,
        big,
        Hq,
        Hk,
        Dh,
        att.snr,
        att.n_noise,
        att.gate_bias,
        T,
        small_off=2 * Hk * Rk + 2 * Hk,
        big_off=Hq * Dh + 2 * Rk * Dh,
    )
    if _GEO_STATS:
        gmix, gmst = gemv([ymix], [L["mixer_proj"]], T=T, stats_x=resid)
    else:
        gmix, gmst = gemv([ymix], [L["mixer_proj"]], T=T)[0], None
    resid, g, gst = _moe_block(L, resid, gmix, T, D, gst=gmst)
    return resid, g, gst


_warmup_started = set()


def _warmup(model, head_fn, rows):
    """Compile the decode kernels of the given batch sizes (throwaway caches)."""
    from .model import ArraysCache, CacheList, KVCache

    for T in rows:
        cache = [
            ArraysCache(1)
            if layer.layer_type == "M"
            else CacheList(KVCache(), ArraysCache(2))
            for layer in model.layers
        ]
        tok = mx.zeros((T, 2), dtype=mx.int32)
        mx.eval(model(tok, cache=cache))
        out = model(tok[:, :1], cache=cache)
        if isinstance(out, tuple):
            out = head_fn(out[0])
        mx.eval(out)


_warmup_started = set()


def maybe_warmup(model, head_fn):
    """Kernels are compiled per row bucket (1 / 2 / 4 / 8). For continuous batching, compile
    them all once up front (in the calling thread: compiling from a second thread while the
    first one evaluates crashed MLX), so that a new batch size does not stall generation.
    OLALA_WARMUP=0 disables it."""
    if (
        os.environ.get("OLALA_WARMUP", "1") != "1"
        or id(model) in _warmup_started
        or not enabled()
    ):
        return
    _warmup_started.add(id(model))
    _warmup(model, head_fn, (1, 2, 4, 8))


def decode(model, x, cache, positions):
    """One fused decode step. x: (B, 1, D) f32 embeddings (not yet normalized).
    Returns the final residual (B, 1, D) f32; the caller applies the head (see head())."""
    plan = _plan(model)
    B, _, D = x.shape
    T = B
    pos = positions.reshape(T).astype(mx.int32)
    shared = {}
    resid = x.reshape(
        T, D
    )  # residual stream (T, D) f32 (embedding before the first layer)
    g = None  # pending update for the geodesic of the previous sub-layer
    prev_geo, prev_li = None, 0
    gst = (
        None  # geodesic sums of the pending update g (prologue "geos"), when available
    )
    for L in plan.layers:
        li = L["li"]
        if L["type"] == "M":
            c = cache[li]
            resid, g, c[0], gst = _m_layer(
                L, resid, g, prev_geo, prev_li, c[0], T, D, gst
            )
        else:
            resid, g, gst = _v_layer(
                L, resid, g, prev_geo, prev_li, cache[li], pos, T, D, shared, gst
            )
        prev_geo, prev_li = L["geo_mlp"], li
    return geodesic(resid, g, prev_geo, prev_li, T, D).reshape(B, 1, D)


def head(lm_head, x):
    """lm_head on f32 activations. x: (B, 1, D)."""
    B, _, D = x.shape
    return gemv([x.reshape(B, D)], [W(lm_head)], T=B)[0].reshape(B, 1, -1)
