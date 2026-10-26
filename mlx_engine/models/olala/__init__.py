"""Olala model support (OlalaForCausalLM / model_type "olala").

``register()`` makes the model importable as ``mlx_lm.models.olala`` so that
mlx-lm's loader finds it like any built-in architecture.
"""

import sys


def register():
    from . import model

    sys.modules.setdefault("mlx_lm.models.olala", model)
