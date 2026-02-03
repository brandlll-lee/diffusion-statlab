import torch

from src.schedules import get_schedule


def test_schedule_alpha_sigma_constraints():
    alpha_min = 0.1
    num_steps = 50

    for schedule_type in ["circular", "cosine"]:
        alphas, sigmas, stats = get_schedule(
            schedule_type, num_steps, alpha_min, return_stats=True
        )
        assert torch.all(alphas >= alpha_min - 1e-8)
        assert torch.allclose(alphas**2 + sigmas**2, torch.ones_like(alphas), atol=1e-6)
        assert stats is not None
        assert stats.clamp_ratio >= 0.0
