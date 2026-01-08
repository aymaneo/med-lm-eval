#!/bin/bash
# RunPod Setup Script for RL Training
# Sets up Python environment, prime-rl, and dependencies

set -e  # Exit on error

echo "========================================"
echo "RL Training - RunPod Setup"
echo "========================================"
echo ""

# ==============================================================================
# 1. Check GPU
# ==============================================================================
echo "[1/5] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU found"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ==============================================================================
# 2. Install uv Package Manager
# ==============================================================================
echo "[2/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed"
else
    echo "uv already installed"
fi

# Add uv to PATH permanently
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Verify uv is accessible
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    exit 1
fi
echo ""

# ==============================================================================
# 3. Setup Main Environment
# ==============================================================================
echo "[3/5] Setting up main environment..."

# Get repo root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Create main venv
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate

# Install base dependencies
uv pip install 'verifiers[rl]'
uv pip install -e .

echo "Main environment ready"
echo ""

# ==============================================================================
# 4. Setup prime-rl
# ==============================================================================
echo "[4/5] Setting up prime-rl..."

cd "$REPO_ROOT/rl_training"

# Clone if not exists
if [ ! -d "prime-rl" ]; then
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git
fi

cd prime-rl
git pull

# Create prime-rl venv
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Install prime-rl and medqa environment
uv pip install -e .
uv pip install -e "$REPO_ROOT/environments/medqa"

# Install flash-attention (required by trainer)
echo "Installing flash-attention (may take 2-3 minutes)..."
uv pip install flash-attn --no-build-isolation

echo "prime-rl ready"
echo ""

# ==============================================================================
# 5. Verify Installation
# ==============================================================================
echo "[5/5] Verifying installation..."

python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  cd $REPO_ROOT/rl_training/prime-rl"
echo "  source .venv/bin/activate"
echo "  export WANDB_API_KEY=your-key"
echo "  uv run rl @ $REPO_ROOT/rl_training/configs/medqa-baseline.toml"
echo ""
