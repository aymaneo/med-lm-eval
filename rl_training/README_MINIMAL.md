# RL Training Setup

## Quick Start on RunPod

```bash
cd med-lm-eval/rl_training
bash scripts/setup_runpod.sh
source ../.venv/bin/activate
uv run prime-rl @ configs/medqa-baseline.toml
```

## Monitor

- Disk: `df -h /workspace`
- GPU: `nvidia-smi`
- Wandb: https://wandb.ai

## Evaluate

```bash
python scripts/evaluate.py experiments/medqa-baseline
```
