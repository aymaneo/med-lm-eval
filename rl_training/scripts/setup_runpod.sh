#!/bin/bash
# RunPod Setup Script for MedARC RL Training
# This script sets up everything you need to train on RunPod

set -e  # Exit on error

echo "========================================="
echo "MedARC RL Training - RunPod Setup"
echo "========================================="
echo ""

# ==============================================================================
# 1. System Check
# ==============================================================================
echo "[1/7] Checking system..."

# Check if we're on a CUDA-capable machine
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check CUDA version
echo "CUDA Version:"
nvidia-smi | grep "CUDA Version"

echo ""

# ==============================================================================
# 2. Install uv (fast Python package manager)
# ==============================================================================
echo "[2/7] Installing uv package manager..."

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv installed"
else
    echo "✅ uv already installed"
fi

echo ""

# ==============================================================================
# 3. Create Python Environment
# ==============================================================================
echo "[3/7] Setting up Python environment..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate
echo "✅ Virtual environment activated"

echo ""

# ==============================================================================
# 4. Install Dependencies
# ==============================================================================
echo "[4/7] Installing dependencies..."

# Install verifiers with RL extras
echo "Installing verifiers[rl]..."
uv pip install 'verifiers[rl]'

# Install flash-attention for faster training
echo "Installing flash-attention (this may take a few minutes)..."
uv pip install flash-attn --no-build-isolation

# Install your custom medarc_verifiers package
echo "Installing medarc_verifiers..."
cd ..  # Go to repo root
uv pip install -e .
cd rl_training

echo "✅ Dependencies installed"
echo ""

# ==============================================================================
# 5. Setup prime-rl
# ==============================================================================
echo "[5/7] Setting up prime-rl..."

# Run verifiers setup to install prime-rl
uv run vf-setup

echo "✅ prime-rl installed"
echo ""

# ==============================================================================
# 6. Configure Wandb (Optional but recommended)
# ==============================================================================
echo "[6/7] Configuring Weights & Biases..."

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set"
    echo "To enable wandb logging:"
    echo "1. Get API key from https://wandb.ai/authorize"
    echo "2. Run: export WANDB_API_KEY=your-key"
    echo "3. Run: wandb login"
    echo ""
    echo "Skipping wandb setup for now..."
else
    wandb login
    echo "✅ Wandb configured"
fi

echo ""

# ==============================================================================
# 7. Verify Installation
# ==============================================================================
echo "[7/7] Verifying installation..."

# Check Python packages
echo "Checking installed packages:"
uv pip list | grep -E "verifiers|transformers|torch" || true

# Check if CUDA is available in PyTorch
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify your config: cat configs/medqa-baseline.toml"
echo "2. Start training: uv run prime-rl @ configs/medqa-baseline.toml"
echo "3. Monitor in wandb: https://wandb.ai"
echo ""
echo "Estimated training time: 2-3 hours"
echo "Estimated cost: ~$1.20 on RTX 4090"
echo ""
echo "Happy training! 🚀"
