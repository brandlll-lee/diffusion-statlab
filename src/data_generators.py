"""
数据生成器实现。

提供两种 x 生成模式：
A) 高维高斯：x ~ N(0, I)，可选归一化到 unit norm
B) 流形/低维子空间模拟：x = A @ u，其中 u ~ N(0, I)，A ∈ R^{d×k}

以及 epsilon 生成：ε ~ N(0, I)，可选归一化
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


class DataGenerator:
    """数据生成器，支持高维高斯和流形模式。"""

    def __init__(
        self,
        d: int,
        x_mode: str = "gaussian",
        manifold_k: Optional[int] = None,
        normalize_x: bool = True,
        normalize_eps: bool = False,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        初始化数据生成器。

        Args:
            d: 数据维度
            x_mode: 'gaussian' 或 'manifold'
            manifold_k: 流形模式下的子空间维度（仅当 x_mode='manifold' 时需要）
            normalize_x: 是否将 x 归一化到 unit norm
            normalize_eps: 是否将 epsilon 归一化到 unit norm
            seed: 随机种子
            device: 'cpu' 或 'cuda'

        Raises:
            ValueError: 参数不合法
        """
        self.d = d
        self.x_mode = x_mode
        self.manifold_k = manifold_k
        self.normalize_x = normalize_x
        self.normalize_eps = normalize_eps
        self.device = device

        # 验证参数
        if x_mode not in ("gaussian", "manifold"):
            raise ValueError(f"Unknown x_mode: '{x_mode}'. Use 'gaussian' or 'manifold'.")

        if x_mode == "manifold":
            if manifold_k is None:
                raise ValueError("manifold_k must be specified when x_mode='manifold'")
            if manifold_k > d:
                raise ValueError(f"manifold_k ({manifold_k}) cannot exceed d ({d})")

        # 设置随机种子（用于生成投影矩阵 A）
        if seed is not None:
            torch.manual_seed(seed)

        # 流形模式：生成固定的投影矩阵 A
        self._projection_matrix: Optional[Tensor] = None
        if x_mode == "manifold" and manifold_k is not None:
            self._projection_matrix = self._create_projection_matrix(d, manifold_k)

    def _create_projection_matrix(self, d: int, k: int) -> Tensor:
        """
        创建投影矩阵 A ∈ R^{d×k}。

        使用正交化的随机矩阵，保证投影后的方差一致。

        Args:
            d: 输出维度
            k: 输入维度（子空间维度）

        Returns:
            A: shape (d, k)
        """
        # 随机初始化
        A = torch.randn(d, k, device=self.device)

        # QR 分解得到正交基（仅取前 k 列）
        Q, _ = torch.linalg.qr(A)

        # 缩放使得 E[||Ax||^2] = k（当 x~N(0,I) 时）
        # 由于 Q 的列是正交单位向量，||Qx||^2 = ||x||^2
        return Q

    def generate_x(self, n: int, seed: Optional[int] = None) -> Tensor:
        """
        生成数据 x。

        Args:
            n: 样本数量
            seed: 随机种子（可选）

        Returns:
            x: shape (n, d)
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.x_mode == "gaussian":
            x = self._generate_gaussian_x(n)
        else:  # manifold
            x = self._generate_manifold_x(n)

        if self.normalize_x:
            x = self._normalize_to_unit_norm(x)

        return x

    def _generate_gaussian_x(self, n: int) -> Tensor:
        """
        生成高维高斯 x ~ N(0, I)。

        Args:
            n: 样本数量

        Returns:
            x: shape (n, d)
        """
        return torch.randn(n, self.d, device=self.device)

    def _generate_manifold_x(self, n: int) -> Tensor:
        """
        生成流形模式的 x = A @ u，其中 u ~ N(0, I)。

        Args:
            n: 样本数量

        Returns:
            x: shape (n, d)
        """
        assert self._projection_matrix is not None
        assert self.manifold_k is not None

        # u ~ N(0, I_k)
        u = torch.randn(n, self.manifold_k, device=self.device)

        # x = u @ A.T，即 x_i = A @ u_i
        x = u @ self._projection_matrix.T

        return x

    def generate_epsilon(self, n: int, seed: Optional[int] = None) -> Tensor:
        """
        生成噪声 ε ~ N(0, I)。

        Args:
            n: 样本数量
            seed: 随机种子（可选）

        Returns:
            eps: shape (n, d)
        """
        if seed is not None:
            torch.manual_seed(seed)

        eps = torch.randn(n, self.d, device=self.device)

        if self.normalize_eps:
            eps = self._normalize_to_unit_norm(eps)

        return eps

    def generate_batch(
        self,
        n: int,
        seed_x: Optional[int] = None,
        seed_eps: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        同时生成 x 和 epsilon。

        Args:
            n: 样本数量
            seed_x: x 的随机种子
            seed_eps: epsilon 的随机种子

        Returns:
            (x, eps): 两个 shape 均为 (n, d) 的张量
        """
        x = self.generate_x(n, seed=seed_x)
        eps = self.generate_epsilon(n, seed=seed_eps)
        return x, eps

    @staticmethod
    def _normalize_to_unit_norm(tensor: Tensor) -> Tensor:
        """
        将张量的每行归一化到 unit norm。

        Args:
            tensor: shape (n, d)

        Returns:
            normalized: shape (n, d)，每行的 L2 范数为 1
        """
        norm = tensor.norm(dim=-1, keepdim=True)
        # 避免除零
        norm = torch.clamp(norm, min=1e-8)
        return tensor / norm

    @property
    def projection_matrix(self) -> Optional[Tensor]:
        """获取投影矩阵（仅流形模式）。"""
        return self._projection_matrix

    def get_config(self) -> dict:
        """返回生成器配置。"""
        return {
            "d": self.d,
            "x_mode": self.x_mode,
            "manifold_k": self.manifold_k,
            "normalize_x": self.normalize_x,
            "normalize_eps": self.normalize_eps,
            "device": self.device,
        }
