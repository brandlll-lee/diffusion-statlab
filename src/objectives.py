"""
目标函数与统计量的基础计算。
"""

from typing import Tuple

import torch
from torch import Tensor


def _assert_same_shape(x: Tensor, y: Tensor, name_x: str, name_y: str) -> None:
    if x.shape != y.shape:
        raise ValueError(f"{name_x} and {name_y} must have the same shape.")
    if x.device != y.device:
        raise ValueError(f"{name_x} and {name_y} must be on the same device.")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise ValueError(f"{name_x} and {name_y} must be floating point tensors.")


def compute_v(x: Tensor, eps: Tensor, alpha: float, sigma: float) -> Tensor:
    """
    计算 v-prediction target: v = α * ε - σ * x。

    Args:
        x: shape (n, d)
        eps: shape (n, d)
        alpha: scalar
        sigma: scalar

    Returns:
        v: shape (n, d)
    """
    _assert_same_shape(x, eps, "x", "eps")
    return alpha * eps - sigma * x


def compute_eps_scaled(
    eps: Tensor, alpha: float, sigma: float, alpha_min: float
) -> Tuple[Tensor, float]:
    """
    计算 scaled epsilon target: eps_scaled = (σ/α) * ε。

    Args:
        eps: shape (n, d)
        alpha: scalar
        sigma: scalar
        alpha_min: alpha 的最小值（数值稳定）

    Returns:
        eps_scaled: shape (n, d)
        alpha_safe: 实际使用的 alpha
    """
    alpha_safe = max(float(alpha), float(alpha_min))
    scale = sigma / alpha_safe
    return scale * eps, alpha_safe


def compute_norm_squared(tensor: Tensor) -> Tensor:
    """
    计算平方范数 ||x||^2。

    Args:
        tensor: shape (n, d)

    Returns:
        norm_sq: shape (n,)
    """
    return (tensor**2).sum(dim=-1)


def compute_cross_term(x: Tensor, eps: Tensor) -> Tensor:
    """
    计算 cross-term: ε·x（点积）。

    Args:
        x: shape (n, d)
        eps: shape (n, d)

    Returns:
        cross: shape (n,)
    """
    _assert_same_shape(x, eps, "x", "eps")
    return (x * eps).sum(dim=-1)
