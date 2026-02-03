import math

from src.stats import DimensionSweepCalculator


def test_dimension_sweep_expected_std_matches_normalization():
    calc = DimensionSweepCalculator(alpha_min=1e-4)
    dims = [64, 256]

    results = calc.compute_cross_term_vs_dimension(
        dimensions=dims,
        num_samples=512,
        normalize_x=True,
        normalize_eps=True,
        seed=123,
    )

    for r in results:
        assert math.isclose(r["expected_std"], 1.0 / math.sqrt(r["d"]), rel_tol=1e-6)
