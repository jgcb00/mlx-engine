# Olala (OlalaForCausalLM) in mlx-engine

Olala 7A1B: 29 Mamba-3 MIMO layers and 7 differential tensor-product attention
layers, a 256-expert MoE (6 active, latent 384) on every layer, and a geodesic
(rotation) residual update. The model is implemented here because mlx-lm /
mlx-vlm do not ship it; `register()` (called by `mlx_engine.register_models`)
exposes it as `mlx_lm.models.olala` and `mlx_vlm.models.olala`, so both the
sequential `ModelKit` and the continuous-batching `BatchedVisionModelKit` load it
like a built-in architecture.

## Converting a checkpoint

```bash
python -m mlx_engine.models.olala.convert \
    --hf-path <hf checkpoint dir> --mlx-path <out dir> --recipe q6
```

| recipe | content | size | KLD vs llama.cpp bf16 |
|---|---|---|---|
| `bf16` | no quantization | 13 GiB | 0.004 |
| `q8` | 8-bit, group 64 | 6.8 GiB | 0.007 |
| `q6` | 6-bit, group 64 | 5.2 GiB | 0.019 |
| `q5e6` | experts 5-bit, rest 6-bit | 4.6 GiB | 0.022 |
| `q4e8` | experts 4-bit / group 32, rest 8-bit | 4.5 GiB | 0.032 |
| `q4` | 4-bit, group 64 | 3.6 GiB | 0.18 (not recommended) |

KLD: mean over 4 x 255 wikitext-2 positions against the log-probs of the
llama.cpp bf16 GGUF (`llama-perplexity --kl-divergence-base`). The router and the
small per-head projections stay unquantized. Projections that read the same input
are merged at load time (`Model.sanitize`) and saved merged; the converted
checkpoint does not need the HF remote code.

## Performance (Mac Studio M3 Ultra, 819 GB/s)

| format | bytes / token | decode B=1 | share of 819 GB/s | B=4 total | B=64 total | prefill (8k, 2048 chunks) |
|---|---|---|---|---|---|---|
| bf16 | 2.58 GB | 191 tok/s | 60 % | 545 tok/s | 1850 tok/s | 4750 tok/s |
| q8 | 1.45 GB | 266 tok/s | 47 % | 535 tok/s | 1754 tok/s | 4840 tok/s |
| q6 | 1.14 GB | 294 tok/s | 41 % | 570 tok/s | 1722 tok/s | 4790 tok/s |

Decode B=1 after a longer prompt (q6 / bf16): 293 / 191 tok/s at 64 tokens of context,
275 / 183 at 1k, 270 / 180 at 2.5k, 241 / 168 at 8k.

For reference, llama.cpp (same machine, Metal) decodes one stream at 149 tok/s
(bf16) and 178 tok/s (q8_0-safe2). Bytes / token counts every weight read for
one decoded token (6 of 256 experts per layer, the head) plus the Mamba state read
and write. Through mlx-engine's `BatchedVisionModelKit`, 4 and 8 concurrent
requests (q6) reach ~405 tok/s aggregate.

## How decoding works (`metal.py`)

A decode step (one token per row, up to 128 rows) runs as a short chain of fused
`mx.fast.metal_kernel`s instead of the ~60 small ops per layer of the reference
path (`kernels.py`, kept for prefill fallback and testing):

- M layer: GEMV(in_proj, with the geodesic update of the previous MoE as
  prologue) -> Mamba-3 step (state in registers, mixer_proj fused at B=1) ->
  GEMV(router + MoE down + shared fc_1, geodesic prologue) -> routed experts
  (top-k in-kernel) -> GEMV(up_proj + shared fc_2)
- V layer: GEMV(all projections) -> q/k/v prep with in-place KV append ->
  flash-decoding attention with soft-capping -> combine (differential heads,
  output gate) -> GEMV(mixer_proj) -> MoE as above

Activations stay in f32; weights are read as bf16 / f16 / f32 or MLX affine
4 / 5 / 6 / 8-bit. Above 8 rows (bf16) / 5 rows (quantized) the matrix products
use MLX's (q)matmul. Prefill uses MLX matmuls, a chunked Mamba-3 kernel
(`m3_chunked`: chunks of 4 tokens, the state stays in registers as 8x8
simdgroup matrices, ~3x faster than the sequential `m3_prefill`), a
flash-attention kernel with soft-capping on simdgroup matrices
(`attn_prefill_mma`, ~2.5x faster than the scalar `attn_prefill`) and the fused
geodesic kernel. At 8k tokens (q6) the prefill time splits into Mamba-3 ~24 %,
MoE ~23 %, attention ~20 %, dense projections most of the rest.

Decode attention uses `attn_decode_short` up to 1024 keys (lane = 4 dims, groups of 8 keys
reduced with one butterfly reduce-scatter: latency-bound contexts) and a key-per-lane
kernel beyond.

Things measured along the way that shape the code:

- Binding a large array to a kernel counts its full size toward MLX's per
  command-buffer budget (`MLX_MAX_MB_PER_BUFFER`) and forced a commit almost every
  kernel: weights are bound through a 16-element view (`_handle`).
- mlx-lm's KV append (slice update) copies the whole cache every token while the
  previous command buffer is in flight; `InplaceAppendMixin` appends in place when
  no one else can see the positions written (copy-on-write otherwise).
- Kernels are compiled per row bucket (1 / 2 / 4 / 8); the mlx-vlm wrapper
  compiles them all on its first call (~1.6 s at load), so a new batch size in a
  server does not stall on Metal compilation (compiling from a background thread
  while another thread evaluates crashed MLX).
- The GEMV producing a residual update also accumulates the geodesic sums
  (x.x, x.g, g.g), so the next GEMV's prologue needs no reduction.
- A factored (TPA) KV cache (the rank-4 B_k / B_v rows plus per-head
  coefficients, 2.8 KB instead of 6.1 KB per token and layer) was prototyped for
  decoding: exact, but the attention kernel was 1.3-2.9x slower. It needs ~4.6x
  the multiply-adds of the full cache (all 48 heads against 4 rows per key, token
  shift included), which at the M3 Ultra's ~7 TFMA/s (simdgroup matrices) and
  819 GB/s is at best on par with reading the full cache.
- A decode step is the sum of the latencies of its ~230 dependent kernels (~3.5 us dispatch
  floor each on M3 Ultra, plus the first-load latency); kernels that depend on nothing
  (e.g. zeroing a buffer) overlap and cost nothing. Removing 72 of them by moving the work
  into the consumers made decoding 6 % slower.
- Scaling a simdgroup-matrix accumulator through `thread_elements()` between products made
  the prefill attention kernel ~8x slower; multiplying by a diagonal matrix instead keeps it
  fast (the Mamba-3 kernel was not affected).
- GEMV rows heavier than 4 KB (the f32 router next to the quantized `in_cat`) are split
  across simdgroups: one heavy row per simdgroup made those threadgroups finish last.
- Routed-expert outputs are summed with float atomics (fastest). The summation
  order then varies between runs; `OLALA_DETERMINISTIC=1` uses fixed-order partial
  sums instead (~3 % slower).

Switches: `OLALA_FUSED=0` / `OLALA_FUSED_PREFILL=0` use the reference path for
decode / prefill, `OLALA_M3_CHUNKED=0` the sequential Mamba-3 prefill kernel,
`OLALA_FA_MMA=0` the scalar prefill attention kernel, `OLALA_MLX_REF=1` disables all Metal
kernels, `OLALA_WARMUP=0` disables the kernel warm-up.

MLX commits a command buffer every few dozen operations by default; with
`MLX_MAX_OPS_PER_BUFFER=400` set before MLX is imported, decoding is ~2 % faster
(application-wide setting, not set here).

## Tests

`tests/test_olala.py` (tiny random model, no checkpoint): fused decode against
the reference path with f32 activations (bf16 and 8 / 6 / 5 / 4-bit), chunked
prefill against token-by-token, chunked against sequential Mamba-3 kernel, split GEMV
rows, MMA against scalar prefill attention, short against long-context decode attention,
fused MoE against the reference experts, cache layout, in-place KV semantics, weight
merging.
