"""
绘图工具函数。
"""

from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_v_norm_vs_t(
    results: List[Dict],
    output_path: str,
    title: str,
    theory_value: float = 1.0,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    t_values = [r["t_value"] for r in results]
    v_means = [r["v_norm_sq_mean"] for r in results]
    v_stds = [r["v_norm_sq_std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values, v_means, linewidth=2, label="E[||v||²]")
    ax.fill_between(
        t_values,
        [m - s for m, s in zip(v_means, v_stds)],
        [m + s for m, s in zip(v_means, v_stds)],
        alpha=0.3,
        label="±1 std",
    )
    ax.axhline(y=theory_value, linestyle="--", alpha=0.5, label=f"Theory (={theory_value})")

    ax.set_xlabel("t (normalized time)", fontsize=12)
    ax.set_ylabel("E[||v||²]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eps_scaled_vs_t(
    results: List[Dict],
    output_path: str,
    d: int,
    normalize_eps: bool,
    log_scale: bool,
    title: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    t_values = [r["t_value"] for r in results]
    eps_means = [r["eps_scaled_norm_sq_mean"] for r in results]

    eps_norm_factor = 1.0 if normalize_eps else float(d)
    theory = []
    for r in results:
        alpha = r["alpha"]
        sigma = r["sigma"]
        theory.append((sigma / alpha) ** 2 * eps_norm_factor)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values, eps_means, linewidth=2, label="E[||eps_scaled||²]")

    if log_scale:
        ax.set_yscale("log")

    if normalize_eps:
        theory_label = "Theory: (σ/α)²"
    else:
        theory_label = f"Theory: (σ/α)² × {d}"
    ax.plot(t_values, theory, linestyle="--", alpha=0.7, label=theory_label)

    ax.set_xlabel("t (normalized time)", fontsize=12)
    ax.set_ylabel("E[||eps_scaled||²]" + (" (log scale)" if log_scale else ""), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_term_vs_t(
    results: List[Dict],
    output_path: str,
    title: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    t_values = [r["t_value"] for r in results]
    cross_means = [r["cross_term_mean"] for r in results]
    cross_stds = [r["cross_term_std"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t_values, cross_means, linewidth=2)
    ax1.axhline(y=0, linestyle="--", alpha=0.5)
    ax1.set_ylabel("E[ε·x]", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_values, cross_stds, linewidth=2)
    ax2.set_xlabel("t (normalized time)", fontsize=12)
    ax2.set_ylabel("Std[ε·x]", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_term_vs_dimension(
    results: List[Dict],
    output_path: str,
    title: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    dimensions = [r["d"] for r in results]
    stds = [r["cross_term_std"] for r in results]
    expected = [r.get("expected_std", None) for r in results]
    expected_label = results[0].get("expected_std_label", "Theory") if results else "Theory"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(dimensions, stds, "o-", linewidth=2, markersize=8, label="Measured std(ε·x)")
    if None not in expected:
        ax.loglog(dimensions, expected, "s--", linewidth=2, markersize=8, label=expected_label)

    ax.set_xlabel("Dimension d", fontsize=12)
    ax.set_ylabel("std(ε·x)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_manifold_v_stability(
    results: Dict,
    output_path: str,
    title: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in results["k_values"]:
        k_data = results["stats_by_k"][k]
        t_values = [s["t_value"] for s in k_data["step_stats"]]
        v_means = [s["v_norm_sq_mean"] for s in k_data["step_stats"]]
        label = f"k={k} (k/d={k_data['k_over_d_ratio']:.3f})"
        ax.plot(t_values, v_means, linewidth=2, label=label)

    ax.axhline(y=1.0, linestyle="--", alpha=0.5, label="Theory (=1)")
    ax.set_xlabel("t (normalized time)", fontsize=12)
    ax.set_ylabel("E[||v||²]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
