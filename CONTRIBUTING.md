# Contributing

Thanks for your interest in improving this project.

## Development Setup

```bash
# 1. 克隆仓库
git clone https://github.com/brandlll-lee/diffusion-statlab.git
cd diffusion-statlab

# 2. 可编辑模式安装（必须，否则测试会失败）
pip install -e .

# 3. 安装开发依赖
pip install pytest
```

## Run Tests

```bash
pytest -v
```

确保所有测试通过后再提交 PR。

## Pull Requests

1. Keep changes focused and well-tested.
2. Update `README.md` when behavior or outputs change.
3. Ensure new experiments are configurable and reproducible.

## Reporting Issues

Please include:

1. Reproduction steps
2. Expected behavior
3. Actual behavior
4. Environment info (`environment.json` from a run, if available)
