#!/bin/bash
# ============================================================================
# Setup Script for Diffusion Model Experiments on NTU HPC
# Run this ONCE to create the conda environment
# ============================================================================
#
# IMPORTANT: Run this on the HEAD NODE (CCDS-TC1), not on compute nodes
#
# Usage:
#   bash setup_hpc_env.sh
#
# ============================================================================

ENV_NAME="diffusion_exp"

echo "=============================================="
echo "Setting up conda environment: $ENV_NAME"
echo "=============================================="

# Load anaconda module
module load anaconda

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Keeping existing environment."
        exit 0
    fi
fi

# Create new environment with Python 3.10 (good compatibility with diffusers)
echo ""
echo "Creating new conda environment..."
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
source activate $ENV_NAME

# Install PyTorch with CUDA support
# Using CUDA 11.8 for compatibility (matches CZ4042_v6)
echo ""
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install diffusers and related packages via pip
# (these are not available via conda)
echo ""
echo "Installing diffusers and ML packages..."
pip install --no-cache-dir \
    diffusers>=0.25.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    safetensors \
    xformers \
    huggingface_hub>=0.20.0

# Install image processing
echo ""
echo "Installing image processing packages..."
pip install --no-cache-dir \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.8.0

# Install utilities
echo ""
echo "Installing utilities..."
pip install --no-cache-dir \
    python-dotenv>=1.0.0 \
    tqdm

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

from diffusers import __version__ as diffusers_version
print(f'Diffusers version: {diffusers_version}')

from transformers import __version__ as transformers_version
print(f'Transformers version: {transformers_version}')

import PIL
print(f'Pillow version: {PIL.__version__}')

import cv2
print(f'OpenCV version: {cv2.__version__}')

print('')
print('All packages installed successfully!')
"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p ~/school_work/fyp/experiments/test_images/content
mkdir -p ~/school_work/fyp/experiments/test_images/inpainting
mkdir -p ~/school_work/fyp/experiments/test_images/style_refs
mkdir -p ~/school_work/fyp/experiments/outputs
mkdir -p ~/school_work/fyp/experiments/logs
mkdir -p ~/.cache/huggingface

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload test images to: ~/school_work/fyp/experiments/test_images/"
echo "   - content/     : Images for img2img and style transfer"
echo "   - inpainting/  : Images and masks for inpainting"
echo ""
echo "2. Submit a job:"
echo "   cd ~/school_work/fyp/experiments"
echo "   sbatch run_experiments_hpc.sh"
echo ""
echo "3. Check job status:"
echo "   squeue -u \$USER"
echo ""
echo "4. View output:"
echo "   tail -f logs/output_diffusion_exp_*.out"
echo ""
