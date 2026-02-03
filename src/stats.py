"""
统计量计算模块。

计算 v-prediction 和 epsilon-prediction 的关键统计量：
- v = α_t * ε - σ_t * x 的平方范数统计
- eps_scaled = (σ_t/α_t) * ε 的平方范数统计
- cross-term ε·x 的统计
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor

from . import objectives


@dataclass
class StepStats:
    """单个时间步的统计结果。"""

    t_idx: int
    t_value: float
    alpha: float
    sigma: float

    # v 统计
    v_norm_sq_mean: float
    v_norm_sq_std: float

    # eps_scaled 统计
    eps_scaled_norm_sq_mean: float
    eps_scaled_norm_sq_std: float

    # cross-term 统计
    cross_term_mean: float
    cross_term_std: float

    # batch 信息（用于加权聚合）
    batch_size: int = 0


class TargetStatsCalculator:
    """目标统计量计算器。"""

    def __init__(self, alpha_min: float = 1e-4):
        """
        初始化计算器。

        Args:
            alpha_min: alpha_t 的最小值（用于数值稳定）
        """
        self.alpha_min = alpha_min

    def compute_v(self, x: Tensor, eps: Tensor, alpha: float, sigma: float) -> Tensor:
        """
        计算 v-prediction target: v = α * ε - σ * x

        Args:
            x: 数据，shape (n, d)
            eps: 噪声，shape (n, d)
            alpha: α_t 标量
            sigma: σ_t 标量

        Returns:
            v: shape (n, d)
        """
        return objectives.compute_v(x, eps, alpha, sigma)

    def compute_eps_scaled(self, eps: Tensor, alpha: float, sigma: float) -> Tensor:
        """
        计算 scaled epsilon target: eps_scaled = (σ/α) * ε

        注意：α 会被 clamp 到 alpha_min 以保证数值稳定。

        Args:
            eps: 噪声，shape (n, d)
            alpha: α_t 标量
            sigma: σ_t 标量

        Returns:
            eps_scaled: shape (n, d)
        """
        eps_scaled, _ = objectives.compute_eps_scaled(eps, alpha, sigma, self.alpha_min)
        return eps_scaled

    @staticmethod
    def assert_finite(tensor: Tensor, name: str) -> None:
        """断言张量中不存在 NaN/Inf。"""
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values detected in {name}.")

    def compute_norm_squared(self, tensor: Tensor) -> Tensor:
        """
        计算每个样本的平方范数 ||x||^2。

        Args:
            tensor: shape (n, d)

        Returns:
            norm_sq: shape (n,)
        """
        return objectives.compute_norm_squared(tensor)

    def compute_cross_term(self, x: Tensor, eps: Tensor) -> Tensor:
        """
        计算 cross-term: ε·x（点积）。

        Args:
            x: 数据，shape (n, d)
            eps: 噪声，shape (n, d)

        Returns:
            cross: shape (n,)，每个样本的点积
        """
        return objectives.compute_cross_term(x, eps)

    def compute_step_stats(
        self,
        x: Tensor,
        eps: Tensor,
        alpha: float,
        sigma: float,
        t_idx: int,
        t_value: float,
    ) -> StepStats:
        """
        计算单个时间步的所有统计量。

        Args:
            x: 数据，shape (n, d)
            eps: 噪声，shape (n, d)
            alpha: α_t
            sigma: σ_t
            t_idx: 时间步索引
            t_value: 归一化时间值 [0, 1]

        Returns:
            StepStats 对象
        """
        # 输入检查
        self.assert_finite(x, "x")
        self.assert_finite(eps, "eps")

        # 计算 v
        v = self.compute_v(x, eps, alpha, sigma)
        self.assert_finite(v, "v")
        v_norm_sq = self.compute_norm_squared(v)
        self.assert_finite(v_norm_sq, "v_norm_sq")

        # 计算 eps_scaled
        eps_scaled = self.compute_eps_scaled(eps, alpha, sigma)
        self.assert_finite(eps_scaled, "eps_scaled")
        eps_scaled_norm_sq = self.compute_norm_squared(eps_scaled)
        self.assert_finite(eps_scaled_norm_sq, "eps_scaled_norm_sq")

        # 计算 cross-term
        cross_term = self.compute_cross_term(x, eps)
        self.assert_finite(cross_term, "cross_term")

        return StepStats(
            t_idx=t_idx,
            t_value=t_value,
            alpha=alpha,
            sigma=sigma,
            v_norm_sq_mean=v_norm_sq.mean().item(),
            v_norm_sq_std=v_norm_sq.std().item(),
            eps_scaled_norm_sq_mean=eps_scaled_norm_sq.mean().item(),
            eps_scaled_norm_sq_std=eps_scaled_norm_sq.std().item(),
            cross_term_mean=cross_term.mean().item(),
            cross_term_std=cross_term.std().item(),
            batch_size=x.shape[0],
        )

    def compute_all_steps(
        self,
        x: Tensor,
        eps: Tensor,
        alphas: Tensor,
        sigmas: Tensor,
        t_values: Tensor,
    ) -> List[StepStats]:
        """
        计算所有时间步的统计量。

        Args:
            x: 数据，shape (n, d)
            eps: 噪声，shape (n, d)
            alphas: α_t 序列，shape (num_steps,)
            sigmas: σ_t 序列，shape (num_steps,)
            t_values: 归一化时间值，shape (num_steps,)

        Returns:
            List[StepStats]，长度为 num_steps
        """
        num_steps = len(alphas)
        results = []

        for i in range(num_steps):
            stats = self.compute_step_stats(
                x=x,
                eps=eps,
                alpha=alphas[i].item(),
                sigma=sigmas[i].item(),
                t_idx=i,
                t_value=t_values[i].item(),
            )
            results.append(stats)

        return results

    @staticmethod
    def aggregate_batched_stats(
        all_batch_stats: List[List[StepStats]],
    ) -> List[Dict[str, float]]:
        """
        聚合多个 batch 的统计结果。

        使用 pooled variance 公式严格合并方差：
        σ²_pooled = [Σ(n_i-1)σ²_i + Σn_i(x̄_i - x̄_pooled)²] / (Σn_i - 1)

        Args:
            all_batch_stats: List of List[StepStats]，外层是 batch，内层是时间步

        Returns:
            聚合后的统计量列表
        """
        if not all_batch_stats:
            return []

        num_steps = len(all_batch_stats[0])

        def weighted_mean(values: List[float], weights: List[int]) -> float:
            """加权平均。"""
            total_weight = sum(weights)
            if total_weight == 0:
                return sum(values) / len(values) if values else 0.0
            return sum(v * w for v, w in zip(values, weights)) / total_weight

        def pooled_std(
            means: List[float],
            stds: List[float],
            batch_sizes: List[int],
        ) -> float:
            """
            计算 pooled standard deviation。

            公式：σ²_pooled = [Σ(n_i-1)σ²_i + Σn_i(x̄_i - x̄_pooled)²] / (Σn_i - 1)

            Args:
                means: 各 batch 的均值
                stds: 各 batch 的标准差
                batch_sizes: 各 batch 的样本数

            Returns:
                pooled standard deviation
            """
            total_n = sum(batch_sizes)
            if total_n <= 1:
                return stds[0] if stds else 0.0

            # pooled mean
            pooled_mean = weighted_mean(means, batch_sizes)

            # 组内方差贡献：Σ(n_i - 1) * σ²_i
            within_var = sum(
                (n - 1) * (s ** 2)
                for n, s in zip(batch_sizes, stds)
                if n > 0
            )

            # 组间方差贡献：Σn_i * (x̄_i - x̄_pooled)²
            between_var = sum(
                n * ((m - pooled_mean) ** 2)
                for n, m in zip(batch_sizes, means)
            )

            # pooled variance
            pooled_var = (within_var + between_var) / (total_n - 1)

            return pooled_var ** 0.5

        aggregated = []
        for step_idx in range(num_steps):
            # 收集该时间步所有 batch 的统计量
            step_stats = [batch[step_idx] for batch in all_batch_stats]

            batch_sizes = [s.batch_size if s.batch_size > 0 else 1 for s in step_stats]
            total_samples = sum(batch_sizes)

            # 加权平均 mean
            v_norm_sq_mean = weighted_mean(
                [s.v_norm_sq_mean for s in step_stats], batch_sizes
            )
            eps_scaled_norm_sq_mean = weighted_mean(
                [s.eps_scaled_norm_sq_mean for s in step_stats], batch_sizes
            )
            cross_term_mean = weighted_mean(
                [s.cross_term_mean for s in step_stats], batch_sizes
            )

            # pooled std
            v_norm_sq_std = pooled_std(
                [s.v_norm_sq_mean for s in step_stats],
                [s.v_norm_sq_std for s in step_stats],
                batch_sizes,
            )
            eps_scaled_norm_sq_std = pooled_std(
                [s.eps_scaled_norm_sq_mean for s in step_stats],
                [s.eps_scaled_norm_sq_std for s in step_stats],
                batch_sizes,
            )
            cross_term_std = pooled_std(
                [s.cross_term_mean for s in step_stats],
                [s.cross_term_std for s in step_stats],
                batch_sizes,
            )

            agg = {
                "t_idx": step_stats[0].t_idx,
                "t_value": step_stats[0].t_value,
                "alpha": step_stats[0].alpha,
                "sigma": step_stats[0].sigma,
                "v_norm_sq_mean": v_norm_sq_mean,
                "v_norm_sq_std": v_norm_sq_std,
                "eps_scaled_norm_sq_mean": eps_scaled_norm_sq_mean,
                "eps_scaled_norm_sq_std": eps_scaled_norm_sq_std,
                "cross_term_mean": cross_term_mean,
                "cross_term_std": cross_term_std,
                "total_samples": total_samples,
            }
            aggregated.append(agg)

        return aggregated


class DimensionSweepCalculator:
    """维度扫描实验计算器。"""

    def __init__(self, alpha_min: float = 1e-4):
        self.alpha_min = alpha_min
        self.stats_calc = TargetStatsCalculator(alpha_min=alpha_min)

    def compute_cross_term_vs_dimension(
        self,
        dimensions: List[int],
        num_samples: int,
        normalize_x: bool = True,
        normalize_eps: bool = False,
        seed: int = 42,
    ) -> List[Dict[str, float]]:
        """
        计算 cross-term 统计量随维度的变化。

        Args:
            dimensions: 维度列表，如 [64, 256, 1024, 4096]
            num_samples: 每个维度的样本数
            normalize_x: 是否归一化 x
            seed: 随机种子

        Returns:
            每个维度的统计结果
        """
        from .data_generators import DataGenerator

        results = []
        for d in dimensions:
            torch.manual_seed(seed)

            gen = DataGenerator(
                d=d,
                x_mode="gaussian",
                normalize_x=normalize_x,
                normalize_eps=normalize_eps,
            )

            x, eps = gen.generate_batch(num_samples)
            cross_term = self.stats_calc.compute_cross_term(x, eps)

            if normalize_x and normalize_eps:
                expected_std = 1.0 / (d**0.5)
                expected_label = "Theory: 1/√d (unit x, unit ε)"
            elif (not normalize_x) and (not normalize_eps):
                expected_std = d**0.5
                expected_label = "Theory: √d (Gaussian x, Gaussian ε)"
            else:
                expected_std = 1.0
                expected_label = "Theory: 1 (mixed normalization)"

            results.append({
                "d": d,
                "cross_term_mean": cross_term.mean().item(),
                "cross_term_std": cross_term.std().item(),
                "cross_term_var": cross_term.var().item(),
                "expected_std": expected_std,
                "expected_std_label": expected_label,
                "normalize_x": normalize_x,
                "normalize_eps": normalize_eps,
            })

        return results


class ManifoldSweepCalculator:
    """流形维度扫描实验计算器。"""

    def __init__(self, alpha_min: float = 1e-4):
        self.alpha_min = alpha_min
        self.stats_calc = TargetStatsCalculator(alpha_min=alpha_min)

    def compute_manifold_sweep(
        self,
        d: int,
        k_values: List[int],
        num_samples: int,
        num_steps: int = 50,
        schedule_type: str = "circular",
        normalize_x: bool = True,
        seed: int = 42,
    ) -> Dict[str, List]:
        """
        固定 d，扫描不同 k 值的流形性对统计量的影响。

        Args:
            d: 数据维度
            k_values: 子空间维度列表
            num_samples: 样本数
            num_steps: 时间步数
            schedule_type: 调度类型
            normalize_x: 是否归一化 x
            seed: 随机种子

        Returns:
            包含各 k 值统计结果的字典
        """
        from .data_generators import DataGenerator
        from .schedules import get_schedule, get_t_values

        results = {
            "k_values": k_values,
            "stats_by_k": {},
        }

        alphas, sigmas, _ = get_schedule(schedule_type, num_steps, self.alpha_min)
        t_values = get_t_values(num_steps)

        for k in k_values:
            torch.manual_seed(seed)

            gen = DataGenerator(
                d=d,
                x_mode="manifold",
                manifold_k=k,
                normalize_x=normalize_x,
                normalize_eps=False,
                seed=seed,
            )

            x, eps = gen.generate_batch(num_samples)

            # 计算所有时间步的统计量
            step_stats = self.stats_calc.compute_all_steps(
                x, eps, alphas, sigmas, t_values
            )

            # 计算 cross-term
            cross_term = self.stats_calc.compute_cross_term(x, eps)

            results["stats_by_k"][k] = {
                "step_stats": [
                    {
                        "t_value": s.t_value,
                        "v_norm_sq_mean": s.v_norm_sq_mean,
                        "v_norm_sq_std": s.v_norm_sq_std,
                    }
                    for s in step_stats
                ],
                "cross_term_mean": cross_term.mean().item(),
                "cross_term_std": cross_term.std().item(),
                "k_over_d_ratio": k / d,
            }

        return results
