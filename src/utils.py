"""
通用工具：配置、复现性、环境信息与输出。
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """设置随机种子并可选启用确定性算法。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_git_info(project_root: Path) -> Dict[str, Optional[str]]:
    """获取 git commit 与 dirty 状态。"""
    git_info: Dict[str, Optional[str]] = {"commit": None, "is_dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["commit"] = commit
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(project_root), stderr=subprocess.DEVNULL
        ).decode("utf-8")
        git_info["is_dirty"] = "yes" if status.strip() else "no"
    except Exception:
        pass
    return git_info


def get_env_info(device: str, project_root: Path) -> Dict[str, Any]:
    """收集环境与版本信息。"""
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except Exception:
        matplotlib_version = None

    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": device,
    }

    if torch.cuda.is_available():
        env.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
        })

    env.update(get_git_info(project_root))
    return env


def write_summary_md(
    path: str | Path,
    config: Dict[str, Any],
    schedule_stats: Dict[str, Any],
    env_info: Dict[str, Any],
    main_results: list[Dict[str, Any]],
    dimension_results: Optional[list[Dict[str, Any]]],
    manifold_results: Optional[Dict[str, Any]],
) -> None:
    """写出实验摘要。"""
    v_means = [r["v_norm_sq_mean"] for r in main_results]
    eps_means = [r["eps_scaled_norm_sq_mean"] for r in main_results]
    cross_means = [r["cross_term_mean"] for r in main_results]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    lines = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Config")
    lines.append("```yaml")
    lines.append(yaml.dump(config, default_flow_style=False, allow_unicode=True).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Schedule")
    lines.append(f"- schedule_type: `{schedule_stats.get('schedule_type')}`")
    lines.append(f"- alpha_min: `{schedule_stats.get('alpha_min')}`")
    lines.append(f"- clamp_count: `{schedule_stats.get('clamp_count')}`")
    lines.append(f"- clamp_ratio: `{schedule_stats.get('clamp_ratio'):.6f}`")
    lines.append("")
    lines.append("## Key Statistics")
    lines.append(f"- mean(E[||v||²]) over t: `{_mean(v_means):.6f}`")
    lines.append(f"- mean(E[||eps_scaled||²]) over t: `{_mean(eps_means):.6f}`")
    lines.append(f"- mean(E[ε·x]) over t: `{_mean(cross_means):.6f}`")

    if dimension_results:
        lines.append("")
        lines.append("## Dimension Sweep")
        for r in dimension_results:
            lines.append(
                f"- d={r['d']}: std(ε·x)={r['cross_term_std']:.6f}, "
                f"expected={r['expected_std']:.6f}"
            )

    if manifold_results:
        lines.append("")
        lines.append("## Manifold Sweep")
        for k in manifold_results["k_values"]:
            k_data = manifold_results["stats_by_k"][k]
            lines.append(
                f"- k={k} (k/d={k_data['k_over_d_ratio']:.3f}): "
                f"cross_term_std={k_data['cross_term_std']:.6f}"
            )

    lines.append("")
    lines.append("## Environment")
    lines.append("```json")
    lines.append(json.dumps(env_info, indent=2, ensure_ascii=False))
    lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
