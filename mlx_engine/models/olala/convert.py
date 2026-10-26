"""Convert an Olala HF checkpoint to MLX (optionally quantized).

    python -m mlx_engine.models.olala.convert --hf-path <dir> --mlx-path <out> [-q --q-bits 8]

Same options as ``mlx_lm.convert``; the model is registered first so that mlx-lm
can build it. Projections are merged at load time (Model.sanitize) and saved merged.
"""

from mlx_engine.models.olala import register


def main():
    register()
    from mlx_lm.convert import main as mlx_lm_convert_main

    mlx_lm_convert_main()


if __name__ == "__main__":
    main()
