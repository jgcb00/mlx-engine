"""Olala model support (OlalaForCausalLM / model_type "olala").

``register()`` makes the model importable as ``mlx_lm.models.olala`` and
``mlx_vlm.models.olala`` so that both loaders find it like a built-in
architecture.
"""

import sys


def register():
    from . import model

    sys.modules.setdefault("mlx_lm.models.olala", model)
    try:
        from . import vlm
    except ImportError:  # mlx-vlm not installed
        return
    sys.modules.setdefault("mlx_vlm.models.olala", vlm)
