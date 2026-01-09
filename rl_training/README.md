Setup simplified. Here are the updated commands for RunPod:

  # 1. Clone repo
  cd /opt
  git clone https://github.com/aymaneo/med-lm-eval.git
  cd med-lm-eval
  git checkout rl-training

  # 2. Run setup script
  cd rl_training
  export PATH="$HOME/.local/bin:$PATH"
  bash scripts/setup_runpod.sh

  # 3. After setup completes, add uv to PATH and set tokens
  export PATH="$HOME/.local/bin:$PATH"
  export HF_TOKEN=your-huggingface-token-here

  # 4. Start training with Qwen2.5-3B-Instruct
  cd /opt/med-lm-eval/rl_training/prime-rl
  source .venv/bin/activate
  export WANDB_API_KEY=your-wandb-key-here
  uv run rl @ /opt/med-lm-eval/rl_training/configs/medqa-baseline.toml