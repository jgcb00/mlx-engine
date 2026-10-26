"""Fused Metal kernels for Olala (mx.fast.metal_kernel)."""

import os


def enabled() -> bool:
    return os.environ.get("OLALA_MLX_REF", "0") != "1" and False


def mamba3_forward(proj, dyn, state, m):
    return None


def softcap_attention(q, k, v, scale, softcap, mask):
    return None
