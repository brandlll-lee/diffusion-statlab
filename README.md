# diffusion-statlab

🔬 Monte Carlo 验证：v-prediction vs epsilon-prediction 目标统计特性

本项目通过 Monte Carlo 方法验证 diffusion models 中两种预测目标的关键统计特性。

[![GitHub](https://img.shields.io/badge/GitHub-diffusion--statlab-blue?logo=github)](https://github.com/brandlll-lee/diffusion-statlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 背景理论

### v-prediction Target

$$v = \alpha_t \cdot \varepsilon - \sigma_t \cdot x$$

**理论预测**：在高维条件下，$E[\|v\|^2] \approx 1$（常数），不随 $t$ 变化。

推导依赖：

- $\|x\|^2 \approx 1$（数据归一化）
- $\|\varepsilon\|^2 \approx d$（噪声的高维集中现象），归一化后 $\|\varepsilon\|^2 \approx 1$
- $x \cdot \varepsilon \approx 0$（高维正交性）
- $\alpha_t^2 + \sigma_t^2 = 1$（圆周参数化）

### epsilon-prediction (Scaled) Target

从重构公式：
$$x = \frac{1}{\alpha_t} z_t - \frac{\sigma_t}{\alpha_t} \varepsilon$$

Scaled target：
$$\text{Target} = \frac{\sigma_t}{\alpha_t} \varepsilon$$

**理论预测**：
$$E[\|\text{Target}\|^2] = \frac{\sigma_t^2}{\alpha_t^2} \cdot E[\|\varepsilon\|^2]$$

当 $t \to T$（信号极弱）时，$\alpha_t \to 0$，方差趋向无穷大。

### Cross-term 正交性

**理论预测**：

- $E[\varepsilon \cdot x] = 0$（由于 $\varepsilon$ 与 $x$ 独立）
- $\text{Var}(\varepsilon \cdot x) \propto 1/d$（当 $x$ 归一化时）

## 项目结构

```
diffusion-statlab/
├── src/
│   ├── __init__.py
│   ├── schedules.py          # 调度器（circular, cosine）
│   ├── data_generators.py    # 数据生成器
│   ├── objectives.py         # 目标函数
│   ├── stats.py              # 统计计算
│   ├── plotting.py           # 绘图工具
│   └── utils.py              # 复现与输出工具
├── experiments/
│   ├── __init__.py
│   └── target_stats.py       # 主实验入口
├── configs/
│   └── target_stats.yaml     # 默认配置
├── tests/
│   ├── test_schedule.py
│   ├── test_targets.py
│   └── test_stats.py
├── outputs/                  # 实验输出（自动生成）
└── README.md
```

## 安装依赖

```bash
pip install torch numpy matplotlib pyyaml pytest
```

## 运行实验

### 基本用法

```bash
# Clone the repository
git clone https://github.com/brandlll-lee/diffusion-statlab.git
cd diffusion-statlab

# Run experiment
python -m experiments.target_stats --config configs/target_stats.yaml
```

### 自定义输出目录

```bash
python -m experiments.target_stats --config configs/target_stats.yaml --output_dir my_outputs
```

## 运行测试

```bash
pytest -q
```

### 修改配置

编辑 `configs/target_stats.yaml` 或创建新的配置文件：

```yaml
# 高维高斯模式
x_mode: gaussian
d: 4096
normalize_x: true

# 或流形模式
x_mode: manifold
manifold_k: 64
```

## 配置项说明

| 参数            | 类型  | 默认值   | 说明                                 |
| --------------- | ----- | -------- | ------------------------------------ |
| `seed`          | int   | 42       | 随机种子                             |
| `d`             | int   | 1024     | 数据维度                             |
| `num_samples`   | int   | 10000    | 总样本数                             |
| `batch_size`    | int   | 1000     | 批大小                               |
| `deterministic` | bool  | true     | 是否启用确定性算法                   |
| `schedule_type` | str   | circular | 调度类型：`circular` 或 `cosine`     |
| `x_mode`        | str   | gaussian | x 生成模式：`gaussian` 或 `manifold` |
| `manifold_k`    | int   | 64       | 流形子空间维度                       |
| `normalize_x`   | bool  | true     | 是否归一化 x                         |
| `normalize_eps` | bool  | true     | 是否归一化 epsilon                   |
| `alpha_min`     | float | 1e-4     | alpha_t 最小值（数值稳定）           |
| `num_steps`     | int   | 100      | 时间步数量                           |
| `log_scale_eps` | bool  | true     | eps_scaled 图使用对数 y 轴           |

**复现 `algorithms/v_prediction.ipynb` 的推荐设置**：`normalize_x=true` 且 `normalize_eps=true`。  
若不归一化 `ε`，理论曲线与统计量都会按维度 `d` 成比例放大。

## 输出说明

每次运行会在 `output_dir` 下创建带时间戳的子目录：

```
outputs/run_20260203_120000/
├── resolved_config.yaml      # 解析后的配置
├── environment.json          # 环境与版本信息
├── main_results.csv          # 主实验结果（CSV）
├── main_results.jsonl        # 主实验结果（JSONL）
├── metrics.jsonl             # 统一指标输出（JSONL）
├── summary.md                # 实验摘要
├── dimension_sweep.csv       # 维度扫描结果
├── manifold_sweep.json       # 流形扫描结果
└── plots/
    ├── v_norm_vs_t.png           # E[||v||²] vs t
    ├── eps_scaled_norm_vs_t.png  # E[||eps_scaled||²] vs t
    ├── eps_scaled_vs_t.png       # 兼容旧命名
    ├── cross_term_vs_t.png       # ε·x 统计量 vs t
    ├── cross_term_vs_dimension.png  # std(ε·x) vs d
    ├── dotprod_std_vs_d.png         # std(ε·x) vs d（规范命名）
    └── manifold_v_stability.png     # 不同 k 下的 ||v||² 稳定性
```

## 预期现象与解读

### 1. v_norm_vs_t.png

**预期**：$E[\|v\|^2]$ 应该接近常数 1，不随 $t$ 显著变化。

- **曲线形态**：近似水平直线，在 $y=1$ 附近
- **验证成功**：曲线波动很小（std 阴影区域窄）
- **如果偏离**：
  - 未归一化 x → 曲线值接近 $d$（维度）
  - 未归一化 eps → 曲线值接近 $d$
  - 流形模式 k << d → 可能在 $t$ 接近 1 时偏离

### 2. eps_scaled_vs_t.png

**预期**：$E[\|(σ/α)ε\|^2]$ 应该随 $t$ 增加而增加，在 $t \to 1$ 时发散。

- **曲线形态**：单调递增，在 $t$ 接近 1 时急剧上升
- **对数 y 轴**：应呈现近似线性增长趋势
- **理论曲线**：
  - `normalize_eps=true`：$(σ_t/α_t)^2$
  - `normalize_eps=false`：$(σ_t/α_t)^2 \times d$
- **为什么这是问题**：训练目标方差不稳定 → 梯度不稳定 → 难以学习

### 3. cross_term_vs_t.png

**预期**：

- 上图：$E[\varepsilon \cdot x] \approx 0$
- 下图：$\text{Std}[\varepsilon \cdot x]$ 应该较小（高维正交性）

- **验证成功**：均值在 0 附近波动，标准差相对均值很小
- **如果偏离**：数据分布不满足独立性假设

### 4. cross_term_vs_dimension.png

**预期**（与归一化方式一致）：

- `normalize_x=true` 且 `normalize_eps=true`：$\text{Std}(\varepsilon \cdot x) \propto 1/\sqrt{d}$
- 其他混合归一化：$\text{Std}(\varepsilon \cdot x) \approx 1$
- `normalize_x=false` 且 `normalize_eps=false`：$\text{Std}(\varepsilon \cdot x) \propto \sqrt{d}$

- **曲线形态**：双对数坐标下应符合理论斜率
- **验证高维正交性**：维度越高，cross-term 越接近 0

### 5. manifold_v_stability.png

**预期**：不同 $k/d$ 比例下 $\|v\|^2$ 的稳定性变化。

- $k=d$（全维）：最稳定，接近 1
- $k << d$（低维流形）：可能出现偏离，特别是在 $t$ 大时
- **物理意义**：真实数据通常位于低维流形上，此实验检验正交假设的鲁棒性

## 常见问题

### Q: 为什么需要 `alpha_min`？

A: 当 $\alpha_t \to 0$ 时，$(σ/α)$ 会趋向无穷大，导致数值溢出。`alpha_min` 设置一个下界保证数值稳定。

### Q: 为什么 `normalize_x` 默认为 true？

A: 理论推导假设 $\|x\|^2 = 1$。归一化确保这一假设成立。在实际应用中，图像数据通常会做某种形式的归一化。

### Q: 流形模式有什么意义？

A: 真实数据（如图像）通常位于高维空间的低维流形上。流形模式模拟这种情况，检验 $\varepsilon \cdot x \approx 0$ 的假设是否依然成立。

### Q: circular 和 cosine schedule 有什么区别？

A:

- **circular**：$\alpha = \cos(\phi), \sigma = \sin(\phi)$，保证 $\alpha^2 + \sigma^2 = 1$ 精确成立
- **cosine**：DDPM 改进版的 schedule，更适合图像生成，也经过归一化处理

### Q: 内存不足怎么办？

A: 减小 `batch_size`。程序会分批计算然后聚合结果。

## 扩展实验

### 对比不同 schedule

```yaml
schedule_type: cosine # 改为 cosine
```

### 测试更高维度

```yaml
d: 16384
batch_size: 500 # 减小批大小避免内存问题
```

### 研究非归一化情况

```yaml
normalize_x: false
normalize_eps: false
```

此时 $E[\|v\|^2] \approx d$，而非 1。

## 参考文献

1. Ho et al., "Denoising Diffusion Probabilistic Models" (arXiv:2006.11239)
2. Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models" (arXiv:2202.00512)
3. Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (arXiv:2102.09672)
