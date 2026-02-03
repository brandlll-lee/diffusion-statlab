"""
Diffusion 调度器实现。

提供两种 schedule：
1. 圆周参数化 (circular): α = cos(φ), σ = sin(φ), φ ∈ [0, π/2]
2. Cosine schedule: 基于 cos^2 的常见 diffusion schedule

所有 schedule 保证 α^2 + σ^2 = 1（归一化）。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScheduleStats:
    """Schedule 统计信息。"""

    schedule_type: str
    num_steps: int
    alpha_min: float
    clamp_count: int
    clamp_ratio: float


class BaseSchedule(ABC):
    """调度器基类。"""

    @abstractmethod
    def get_alpha_sigma(self, num_steps: int, alpha_min: float) -> Tuple[Tensor, Tensor]:
        """
        获取 alpha_t 和 sigma_t 序列。

        Args:
            num_steps: 时间步数量
            alpha_min: alpha_t 的最小值（用于数值稳定）

        Returns:
            alpha_t: shape (num_steps,)，从约 1 递减到 alpha_min
            sigma_t: shape (num_steps,)，从约 0 递增到接近 1
        """
        pass

    @staticmethod
    def normalize_alpha_sigma(alpha: Tensor, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        """
        归一化使得 α^2 + σ^2 = 1。

        Args:
            alpha: 原始 alpha 值
            sigma: 原始 sigma 值

        Returns:
            归一化后的 (alpha, sigma)
        """
        norm = torch.sqrt(alpha**2 + sigma**2)
        return alpha / norm, sigma / norm

    @staticmethod
    def apply_alpha_min(alpha: Tensor, alpha_min: float) -> Tuple[Tensor, Tensor, int]:
        """
        对 alpha 应用下界，同时保持 α^2 + σ^2 = 1。

        Args:
            alpha: shape (num_steps,)
            alpha_min: alpha 的最小值

        Returns:
            alpha_clamped: shape (num_steps,)
            sigma_recomputed: shape (num_steps,)
            clamp_count: 被 clamp 的步数
        """
        alpha_min = float(alpha_min)
        if alpha_min < 0.0 or alpha_min >= 1.0:
            raise ValueError("alpha_min must be in [0, 1).")

        alpha_clamped = torch.clamp(alpha, min=alpha_min)
        clamp_count = int((alpha < alpha_min).sum().item())

        # 根据圆周约束重新计算 sigma，确保 alpha^2 + sigma^2 = 1
        sigma_recomputed = torch.sqrt(torch.clamp(1.0 - alpha_clamped**2, min=0.0))

        return alpha_clamped, sigma_recomputed, clamp_count


class CircularSchedule(BaseSchedule):
    """
    圆周参数化调度器。

    α = cos(φ), σ = sin(φ), φ ∈ [φ_min, π/2]
    其中 φ_min 由 alpha_min 确定：φ_min = arccos(alpha_min)
    """

    def get_alpha_sigma(self, num_steps: int, alpha_min: float) -> Tuple[Tensor, Tensor]:
        """
        生成圆周参数化的 alpha 和 sigma。

        Args:
            num_steps: 时间步数量
            alpha_min: alpha_t 的最小值

        Returns:
            alpha_t: shape (num_steps,)
            sigma_t: shape (num_steps,)
        """
        # φ 从 0 到 arccos(alpha_min)
        phi_max = math.acos(max(alpha_min, 1e-6))
        phi = torch.linspace(0, phi_max, num_steps)

        alpha = torch.cos(phi)
        sigma = torch.sin(phi)

        # 确保 alpha >= alpha_min，且保持 α^2 + σ^2 = 1
        alpha, sigma, _ = self.apply_alpha_min(alpha, alpha_min)

        return alpha, sigma


class CosineSchedule(BaseSchedule):
    """
    Cosine schedule（改进版 DDPM schedule）。

    基于论文 "Improved Denoising Diffusion Probabilistic Models"
    α_bar_t = cos^2((t/T + s) / (1 + s) * π/2)

    经过归一化处理保证 α^2 + σ^2 = 1。
    """

    def __init__(self, s: float = 0.008):
        """
        Args:
            s: offset 参数，防止 t=0 时 beta 过小
        """
        self.s = s

    def get_alpha_sigma(self, num_steps: int, alpha_min: float) -> Tuple[Tensor, Tensor]:
        """
        生成 cosine schedule 的 alpha 和 sigma。

        Args:
            num_steps: 时间步数量
            alpha_min: alpha_t 的最小值

        Returns:
            alpha_t: shape (num_steps,)
            sigma_t: shape (num_steps,)
        """
        t = torch.linspace(0, 1, num_steps)
        s = self.s

        # α_bar_t = cos^2((t + s) / (1 + s) * π/2)
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # 归一化使 α_bar_0 = 1

        # α_t = sqrt(α_bar_t), σ_t = sqrt(1 - α_bar_t)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1 - alpha_bar)

        # 确保 alpha >= alpha_min，且保持 α^2 + σ^2 = 1
        alpha, sigma, _ = self.apply_alpha_min(alpha, alpha_min)

        return alpha, sigma


def get_schedule(
    schedule_type: str,
    num_steps: int,
    alpha_min: float = 1e-4,
    return_stats: bool = False,
    **kwargs,
) -> Tuple[Tensor, Tensor, Optional[ScheduleStats]]:
    """
    工厂函数：获取指定类型的调度器。

    Args:
        schedule_type: 调度类型，'circular' 或 'cosine'
        num_steps: 时间步数量
        alpha_min: alpha_t 的最小值（默认 1e-4）
        return_stats: 是否返回 clamp 统计信息
        **kwargs: 传递给具体调度器的额外参数

    Returns:
        alpha_t: shape (num_steps,)
        sigma_t: shape (num_steps,)
        stats: ScheduleStats 或 None

    Raises:
        ValueError: 未知的调度类型
    """
    schedule_classes = {
        "circular": CircularSchedule,
        "cosine": CosineSchedule,
    }

    if schedule_type not in schedule_classes:
        raise ValueError(
            f"Unknown schedule type: '{schedule_type}'. "
            f"Available types: {list(schedule_classes.keys())}"
        )

    schedule = schedule_classes[schedule_type](**kwargs)
    alpha, sigma = schedule.get_alpha_sigma(num_steps, alpha_min)

    if return_stats:
        clamp_count = int((alpha <= float(alpha_min) + 1e-12).sum().item())
        stats = ScheduleStats(
            schedule_type=schedule_type,
            num_steps=num_steps,
            alpha_min=float(alpha_min),
            clamp_count=clamp_count,
            clamp_ratio=clamp_count / float(num_steps),
        )
        return alpha, sigma, stats

    return alpha, sigma, None


def get_t_values(num_steps: int) -> Tensor:
    """
    获取归一化的时间值 t ∈ [0, 1]。

    Args:
        num_steps: 时间步数量

    Returns:
        t: shape (num_steps,)，从 0 到 1
    """
    return torch.linspace(0, 1, num_steps)
