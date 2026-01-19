#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=360
#SBATCH --job-name=diffusion_exp
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

# ============================================================================
# SLURM Job Script for Diffusion Model Experiments
# NTU CCDS GPU Cluster (TC1)
# ============================================================================
#
# Usage:
#   sbatch run_experiments_hpc.sh                    # Run all experiments
#   sbatch run_experiments_hpc.sh img2img           # Run only img2img
#   sbatch run_experiments_hpc.sh inpainting        # Run only inpainting
#   sbatch run_experiments_hpc.sh style             # Run only style transfer
#
# Before running:
#   1. Setup conda environment (see setup_hpc_env.sh)
#   2. Place test images in test_images/ directory
#   3. Create logs/ directory: mkdir -p logs
#
# ============================================================================

# Parse experiment type from command line argument
EXPERIMENT_TYPE=${1:-all}

echo "=============================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment type: $EXPERIMENT_TYPE"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load anaconda
module load cuda/12.1

# Activate conda environment
# Option 1: Use shared environment (if diffusers is installed)
# source activate CZ4042_v6

# Option 2: Use personal environment (recommended)
source activate diffusion_exp

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify Python and PyTorch
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
echo ""

# Set cache directories to avoid quota issues
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"

# Create cache directories
mkdir -p $HF_HOME $TORCH_HOME $TRANSFORMERS_CACHE

# Run experiments
echo "=============================================="
echo "Starting experiments..."
echo "=============================================="

cd /home/yuquan/school_work/fyp/experiments

if [ "$EXPERIMENT_TYPE" == "all" ]; then
    python run_all_experiments.py \
        --input-dir ./test_images \
        --output-dir ./outputs
elif [ "$EXPERIMENT_TYPE" == "img2img" ]; then
    python run_all_experiments.py \
        --input-dir ./test_images \
        --output-dir ./outputs \
        --experiment img2img
elif [ "$EXPERIMENT_TYPE" == "inpainting" ]; then
    python run_all_experiments.py \
        --input-dir ./test_images \
        --output-dir ./outputs \
        --experiment inpainting
elif [ "$EXPERIMENT_TYPE" == "style" ]; then
    python run_all_experiments.py \
        --input-dir ./test_images \
        --output-dir ./outputs \
        --experiment style
else
    echo "Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Valid options: all, img2img, inpainting, style"
    exit 1
fi

echo ""
echo "=============================================="
echo "Job finished at: $(date)"
echo "=============================================="
