"""Convert an Olala HF checkpoint to MLX (optionally quantized).

    python -m mlx_engine.models.olala.convert --hf-path <dir> --mlx-path <out> [-q --q-bits 8]

Same options as ``mlx_lm.convert``; the model is registered first so that mlx-lm
can build it. Projections are merged at load time (Model.sanitize) and saved merged.
"""

import json
import sys
from pathlib import Path

from mlx_engine.models.olala import register


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
    register()
    from mlx_lm.convert import main as mlx_lm_convert_main

    mlx_lm_convert_main()
    if "--mlx-path" in sys.argv:
        _clean(Path(sys.argv[sys.argv.index("--mlx-path") + 1]))


if __name__ == "__main__":
    main()
