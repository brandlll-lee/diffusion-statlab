"""
Monte Carlo 统计验证实验：v-prediction vs epsilon-prediction target 统计特性。

实验内容：
1. 验证 v-target 的平方范数 E[||v||^2] 随 t 近似常数
2. 验证 epsilon-target（scaled）的平方范数随 t 在 α_t→0 时发散
3. 验证 cross-term E[ε·x] ≈ 0 且 Var(ε·x) 随维度增大而下降

运行方式：
    python -m experiments.target_stats --config configs/target_stats.yaml
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generators import DataGenerator
from src.plotting import (
    plot_cross_term_vs_dimension,
    plot_cross_term_vs_t,
    plot_eps_scaled_vs_t,
    plot_manifold_v_stability,
    plot_v_norm_vs_t,
)
from src.schedules import get_schedule, get_t_values
from src.stats import (
    DimensionSweepCalculator,
    ManifoldSweepCalculator,
    TargetStatsCalculator,
)
from src.utils import (
    ensure_dir,
    get_env_info,
    load_yaml,
    save_json,
    save_jsonl,
    save_yaml,
    set_seed,
    write_summary_md,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_results_csv(results: List[Dict], output_path: str) -> None:
    """保存结果到 CSV 文件。"""
    if not results:
        return

    import csv

    keys = results[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to: {output_path}")


def run_main_experiment(config: Dict[str, Any]) -> tuple[List[Dict], Dict[str, Any]]:
    """
    运行主实验：计算 v 和 eps_scaled 的统计量随 t 的变化。

    Args:
        config: 配置字典

    Returns:
        每个时间步的统计结果列表
    """
    # 提取配置
    seed = config["seed"]
    d = config["d"]
    num_samples = config["num_samples"]
    batch_size = config["batch_size"]
    schedule_type = config["schedule_type"]
    x_mode = config["x_mode"]
    manifold_k = config.get("manifold_k")
    normalize_x = config["normalize_x"]
    normalize_eps = config.get("normalize_eps", False)
    alpha_min = config["alpha_min"]
    num_steps = config.get("num_steps", 100)
    device = config.get("device", "cpu")

    logger.info(f"Running main experiment with d={d}, n={num_samples}, schedule={schedule_type}")
    logger.info(f"x_mode={x_mode}, normalize_x={normalize_x}, alpha_min={alpha_min}")

    # 设置随机种子
    set_seed(seed, deterministic=config.get("deterministic", True))

    # 创建数据生成器
    gen = DataGenerator(
        d=d,
        x_mode=x_mode,
        manifold_k=manifold_k,
        normalize_x=normalize_x,
        normalize_eps=normalize_eps,
        seed=seed,
        device=device,
    )

    # 获取 schedule
    alphas, sigmas, schedule_stats = get_schedule(
        schedule_type, num_steps, alpha_min, return_stats=True
    )
    t_values = get_t_values(num_steps)

    # 统计计算器
    stats_calc = TargetStatsCalculator(alpha_min=alpha_min)

    # 分批计算以节省内存
    num_batches = math.ceil(num_samples / batch_size)
    all_batch_stats = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        # 生成数据（每个 batch 使用不同的随机状态）
        x, eps = gen.generate_batch(current_batch_size)

        # 计算该 batch 的统计量
        batch_stats = stats_calc.compute_all_steps(x, eps, alphas, sigmas, t_values)
        all_batch_stats.append(batch_stats)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            logger.info(f"Processed batch {batch_idx + 1}/{num_batches}")

    # 聚合所有 batch 的结果
    aggregated = stats_calc.aggregate_batched_stats(all_batch_stats)

    # 记录 alpha 是否被 clamp
    alpha_min_val = float(alpha_min)
    for r in aggregated:
        r["alpha_clamped"] = bool(r["alpha"] <= alpha_min_val + 1e-12)

    return aggregated, {
        "schedule_type": schedule_stats.schedule_type if schedule_stats else schedule_type,
        "num_steps": schedule_stats.num_steps if schedule_stats else num_steps,
        "alpha_min": schedule_stats.alpha_min if schedule_stats else float(alpha_min),
        "clamp_count": schedule_stats.clamp_count if schedule_stats else 0,
        "clamp_ratio": schedule_stats.clamp_ratio if schedule_stats else 0.0,
    }


def run_dimension_sweep(config: Dict[str, Any]) -> List[Dict]:
    """
    运行维度扫描实验：观察 cross-term 统计量随维度的变化。

    Args:
        config: 配置字典

    Returns:
        每个维度的统计结果
    """
    seed = config["seed"]
    dimensions = config.get("dimension_sweep", [64, 256, 1024, 4096])
    num_samples = config.get("dimension_sweep_samples", config["num_samples"])
    normalize_x = config["normalize_x"]
    normalize_eps = config.get("normalize_eps", False)
    alpha_min = config["alpha_min"]

    logger.info(f"Running dimension sweep: dimensions={dimensions}")

    calc = DimensionSweepCalculator(alpha_min=alpha_min)
    results = calc.compute_cross_term_vs_dimension(
        dimensions=dimensions,
        num_samples=num_samples,
        normalize_x=normalize_x,
        normalize_eps=normalize_eps,
        seed=seed,
    )

    return results


def run_manifold_sweep(config: Dict[str, Any]) -> Dict:
    """
    运行流形扫描实验：固定 d，观察不同 k 值对统计量的影响。

    Args:
        config: 配置字典

    Returns:
        流形扫描结果字典
    """
    seed = config["seed"]
    d = config.get("manifold_sweep_d", 1024)
    k_values = config.get("manifold_k_values", [4, 16, 64, 256])
    num_samples = config.get("manifold_sweep_samples", config["num_samples"])
    num_steps = config.get("num_steps", 100)
    schedule_type = config["schedule_type"]
    normalize_x = config["normalize_x"]
    alpha_min = config["alpha_min"]

    logger.info(f"Running manifold sweep: d={d}, k_values={k_values}")

    calc = ManifoldSweepCalculator(alpha_min=alpha_min)
    results = calc.compute_manifold_sweep(
        d=d,
        k_values=k_values,
        num_samples=num_samples,
        num_steps=num_steps,
        schedule_type=schedule_type,
        normalize_x=normalize_x,
        seed=seed,
    )

    return results


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo validation of v-prediction vs epsilon-prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/target_stats.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        config_path = args.config

    logger.info(f"Loading config from: {config_path}")
    config = load_yaml(config_path)

    # 设置输出目录
    if args.output_dir:
        config["output_dir"] = args.output_dir

    output_dir = config.get("output_dir", "outputs")
    output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)

    logger.info(f"Output directory: {run_dir}")

    # 保存配置
    save_yaml(config, os.path.join(run_dir, "resolved_config.yaml"))

    # ========== 主实验 ==========
    logger.info("=" * 50)
    logger.info("Running main experiment...")
    main_results, schedule_stats = run_main_experiment(config)
    dim_results: Optional[List[Dict[str, Any]]] = None
    manifold_results: Optional[Dict[str, Any]] = None

    # 保存主实验结果
    save_results_csv(main_results, os.path.join(run_dir, "main_results.csv"))
    save_jsonl(main_results, os.path.join(run_dir, "main_results.jsonl"))

    # 绘图
    log_scale_eps = config.get("log_scale_eps", True)

    plot_v_norm_vs_t(
        main_results,
        os.path.join(plots_dir, "v_norm_vs_t.png"),
        title=f"v-prediction: E[||v||²] vs t (d={config['d']}, {config['schedule_type']})",
    )

    plot_eps_scaled_vs_t(
        main_results,
        os.path.join(plots_dir, "eps_scaled_norm_vs_t.png"),
        d=config["d"],
        normalize_eps=config.get("normalize_eps", False),
        log_scale=log_scale_eps,
        title=f"epsilon-prediction: E[||eps_scaled||²] vs t (d={config['d']})",
    )
    plot_eps_scaled_vs_t(
        main_results,
        os.path.join(plots_dir, "eps_scaled_vs_t.png"),
        d=config["d"],
        normalize_eps=config.get("normalize_eps", False),
        log_scale=log_scale_eps,
        title=f"epsilon-prediction: E[||eps_scaled||²] vs t (d={config['d']})",
    )

    plot_cross_term_vs_t(
        main_results,
        os.path.join(plots_dir, "cross_term_vs_t.png"),
        title=f"Cross-term statistics vs t (d={config['d']})",
    )

    # ========== 维度扫描实验 ==========
    if config.get("run_dimension_sweep", True):
        logger.info("=" * 50)
        logger.info("Running dimension sweep experiment...")
        dim_results = run_dimension_sweep(config)
        save_results_csv(dim_results, os.path.join(run_dir, "dimension_sweep.csv"))

        plot_cross_term_vs_dimension(
            dim_results,
            os.path.join(plots_dir, "cross_term_vs_dimension.png"),
            title=f"Cross-term std vs dimension (normalize_x={config['normalize_x']})",
        )
        plot_cross_term_vs_dimension(
            dim_results,
            os.path.join(plots_dir, "dotprod_std_vs_d.png"),
            title=f"std(ε·x) vs dimension d",
        )

    # ========== 流形扫描实验 ==========
    if config.get("run_manifold_sweep", True):
        logger.info("=" * 50)
        logger.info("Running manifold sweep experiment...")
        manifold_results = run_manifold_sweep(config)

        # 保存流形扫描结果
        serializable = {
            "k_values": manifold_results["k_values"],
            "stats_by_k": {
                str(k): v for k, v in manifold_results["stats_by_k"].items()
            },
        }
        save_json(serializable, os.path.join(run_dir, "manifold_sweep.json"))

        plot_manifold_v_stability(
            manifold_results,
            os.path.join(plots_dir, "manifold_v_stability.png"),
            title=f"Manifold mode: E[||v||²] stability (d={config.get('manifold_sweep_d', 1024)})",
        )

    # ========== 统一指标输出 ==========
    metrics_records: List[Dict[str, Any]] = []
    for r in main_results:
        metrics_records.append({"record_type": "main", **r})
    if dim_results is not None:
        for r in dim_results:
            metrics_records.append({"record_type": "dimension_sweep", **r})
    if manifold_results is not None:
        for k in manifold_results["k_values"]:
            k_data = manifold_results["stats_by_k"][k]
            metrics_records.append({
                "record_type": "manifold_sweep",
                "k": k,
                "k_over_d_ratio": k_data["k_over_d_ratio"],
                "cross_term_mean": k_data["cross_term_mean"],
                "cross_term_std": k_data["cross_term_std"],
            })
    metrics_records.append({"record_type": "schedule_stats", **schedule_stats})
    save_jsonl(metrics_records, os.path.join(run_dir, "metrics.jsonl"))

    # ========== Summary 与环境信息 ==========
    env_info = get_env_info(config.get("device", "cpu"), PROJECT_ROOT)
    save_json(env_info, os.path.join(run_dir, "environment.json"))
    write_summary_md(
        os.path.join(run_dir, "summary.md"),
        config=config,
        schedule_stats=schedule_stats,
        env_info=env_info,
        main_results=main_results,
        dimension_results=dim_results,
        manifold_results=manifold_results,
    )

    logger.info("=" * 50)
    logger.info(f"All experiments completed. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
