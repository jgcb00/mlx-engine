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
| bf16 | 2.58 GB | 183 tok/s | 58 % | 494 tok/s | 1816 tok/s | 2838 tok/s |
| q8 | 1.45 GB | 246 tok/s | 43 % | 505 tok/s | 1766 tok/s | 2857 tok/s |
| q6 | 1.14 GB | 271 tok/s | 38 % | 530 tok/s | 1739 tok/s | 2845 tok/s |

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
use MLX's (q)matmul. Prefill uses MLX matmuls, a fused sequential Mamba-3 kernel
(`m3_prefill`), a flash-attention style kernel with soft-capping
(`attn_prefill`) and the fused geodesic kernel.

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
- Routed-expert outputs are summed with float atomics (fastest). The summation
  order then varies between runs; `OLALA_DETERMINISTIC=1` uses fixed-order partial
  sums instead (~3 % slower).

Switches: `OLALA_FUSED=0` / `OLALA_FUSED_PREFILL=0` use the reference path for
decode / prefill, `OLALA_MLX_REF=1` disables all Metal kernels, `OLALA_WARMUP=0`
disables the background compilation.

## Tests

`tests/test_olala.py` (tiny random model, no checkpoint): fused decode against
the reference path with f32 activations (bf16 and 8 / 6 / 5 / 4-bit), chunked
prefill against token-by-token, cache layout, in-place KV semantics, weight
merging.
