"""Convert an Olala HF checkpoint to MLX.

    python -m mlx_engine.models.olala.convert --hf-path <dir> --mlx-path <out> --recipe q6

Recipes (KLD vs the llama.cpp bf16 reference on wikitext, 2 x 512 tokens):
    bf16   no quantization                                       (0.004 with bf16 activations)
    q8     8-bit, group 64                                        0.006, 6.8 GiB
    q6     6-bit, group 64                                        0.019, 5.2 GiB
    q5e6   routed experts 5-bit, everything else 6-bit            0.021, 4.5 GiB
    q4e8   routed experts 4-bit / group 32, everything else 8-bit 0.030, 4.5 GiB
    q4     4-bit, group 64 (large quality loss on this model)     0.12+
The router and the small per-head projections are never quantized. Projections
are merged at load time (Model.sanitize) and saved merged.
"""

import argparse
import json
from pathlib import Path

from mlx_engine.models.olala import register


def _is_expert(path: str) -> bool:
    return path.endswith((".experts.experts", ".experts.output_experts"))


RECIPES = {
    "q8": lambda p: (8, 64),
    "q6": lambda p: (6, 64),
    "q5e6": lambda p: (5, 64) if _is_expert(p) else (6, 64),
    "q4e8": lambda p: (4, 32) if _is_expert(p) else (8, 64),
    "q4": lambda p: (4, 64),
}


def _clean(mlx_path: Path):
    """The converted checkpoint runs without the HF remote code: drop it, so that
    transformers does not ask for trust_remote_code when loading the tokenizer."""
    cfg_path = mlx_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("auto_map", None)
    cfg_path.write_text(json.dumps(cfg, indent=4))
    for name in ("configuration_olala.py", "modeling_olala.py"):
        (mlx_path / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--recipe", choices=["bf16", *RECIPES], default="bf16")
    args = parser.parse_args()

    register()
    from mlx_lm.convert import convert

    if args.recipe == "bf16":
        convert(args.hf_path, args.mlx_path, dtype="bfloat16")
    else:
        from mlx_engine.models.olala.model import quant_predicate

        recipe = RECIPES[args.recipe]

        def predicate(path, module):
            if not hasattr(module, "to_quantized") or not quant_predicate(path, module):
                return False
            bits, group_size = recipe(path)
            return {"bits": bits, "group_size": group_size, "mode": "affine"}

        bits, group_size = recipe("")
        convert(
            args.hf_path,
            args.mlx_path,
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            dtype="bfloat16",
            quant_predicate=predicate,
        )
    _clean(Path(args.mlx_path))


if __name__ == "__main__":
    main()
