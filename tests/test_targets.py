import torch

from src import objectives


def test_compute_v_and_eps_scaled():
    x = torch.tensor([[1.0, 0.0]])
    eps = torch.tensor([[0.0, 1.0]])
    alpha = 0.6
    sigma = 0.8

    v = objectives.compute_v(x, eps, alpha, sigma)
    assert torch.allclose(v, torch.tensor([[-0.8, 0.6]]))

    eps_scaled, alpha_safe = objectives.compute_eps_scaled(eps, alpha=1e-6, sigma=1.0, alpha_min=1e-4)
    assert alpha_safe == 1e-4
    assert torch.allclose(eps_scaled, torch.tensor([[0.0, 10000.0]]))


def test_norm_and_cross_term():
    x = torch.tensor([[3.0, 4.0]])
    eps = torch.tensor([[2.0, 1.0]])

    norm_sq = objectives.compute_norm_squared(x)
    assert torch.allclose(norm_sq, torch.tensor([25.0]))

    cross = objectives.compute_cross_term(x, eps)
    assert torch.allclose(cross, torch.tensor([10.0]))
