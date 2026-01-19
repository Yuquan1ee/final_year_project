# Running Diffusion Model Experiments on NTU HPC

This guide explains how to run the consolidated diffusion model experiments on the NTU CCDS GPU Cluster (TC1).

## VPN Connection (Required if Off-Campus)

Connect to NTU VPN before accessing the HPC cluster:

```bash
# Using GlobalProtect client
sudo gpclient --fix-openssl connect vpngate-student.ntu.edu.sg
```

You'll be prompted for your NTU credentials.

## SSH Connection

```bash
# Connect to HPC cluster
ssh <username>@10.96.189.11

# Example:
ssh ylee136@10.96.189.11
```

## Transfer Files to HPC

```bash
# Transfer entire experiments directory to HPC
scp -r /path/to/local/experiments/ <username>@10.96.189.11:~/school_work/fyp/

# Example:
scp -r ~/school_work/fyp/experiments/ ylee136@10.96.189.11:~/school_work/fyp/

# Transfer specific test images
scp -r ./test_images/ <username>@10.96.189.11:~/school_work/fyp/experiments/
```

## Quick Start

```bash
# 1. SSH to HPC (after VPN connection)
ssh <username>@10.96.189.11

# 2. Setup environment (one-time)
cd ~/school_work/fyp/experiments
bash setup_hpc_env.sh

# 3. Upload test images (see "Required Images" section below)

# 4. Submit job
sbatch run_experiments_hpc.sh

# 5. Monitor job
squeue -u $USER
tail -f logs/output_diffusion_exp_*.out
```

## Required Images

Create the following directory structure and add your test images:

```
experiments/
└── test_images/
    ├── content/              # For img2img and style transfer
    │   ├── landscape.jpg     # Any photos you want to edit/stylize
    │   ├── portrait.jpg
    │   └── street.jpg
    │
    ├── inpainting/           # For inpainting experiments
    │   ├── photo1.jpg        # Original image
    │   ├── photo1_mask.png   # Mask: white=inpaint, black=keep
    │   ├── photo2.jpg
    │   └── photo2_mask.png
    │
    └── style_refs/           # (Optional) For IP-Adapter style transfer
        └── style_image.jpg   # Reference style image
```

### Image Guidelines

**Content Images (for img2img & style transfer):**
- Any JPG/PNG photos work well
- Recommended: 512x512 or larger (will be resized)
- Good subjects: landscapes, portraits, buildings, objects
- 2-5 images is enough for testing

**Inpainting Images & Masks:**
- Image: Regular photo with an object/region to replace
- Mask: Same dimensions as image
  - **White (255)**: Area to inpaint (replace)
  - **Black (0)**: Area to keep unchanged
- You can create masks in any image editor (Photoshop, GIMP, etc.)
- If no mask is provided, the script will auto-generate a center mask

**Sample Images Sources:**
- Your own photos
- [Unsplash](https://unsplash.com) (free high-quality photos)
- [Pexels](https://www.pexels.com) (free stock photos)

## Experiments Overview

### 1. Image-to-Image Editing (`--experiment img2img`)

**Models tested:**
- **InstructPix2Pix** - Follows natural language edit instructions
- **Stable Diffusion img2img** - Transforms image based on prompt

**Test cases:**
- "make it winter with snow"
- "make it sunset"
- "turn it into a painting"
- Various strength levels (0.5, 0.75)

### 2. Inpainting (`--experiment inpainting`)

**Models tested:**
- **Stable Diffusion Inpainting** - Standard inpainting model

**Test cases:**
- Multiple prompts ("garden with flowers", "calm lake", etc.)
- Center region replacement if no mask provided

### 3. Style Transfer (`--experiment style`)

**Models tested:**
- **SD img2img** - Simple style transfer
- **ControlNet (Canny)** - Structure-preserving style transfer

**Styles tested:**
- oil_painting, watercolor, anime, sketch
- cyberpunk, ghibli, impressionist, 3d_render

## Running Experiments

### Full Test Suite
```bash
sbatch run_experiments_hpc.sh
```

### Specific Experiment
```bash
sbatch run_experiments_hpc.sh img2img      # Only img2img
sbatch run_experiments_hpc.sh inpainting   # Only inpainting
sbatch run_experiments_hpc.sh style        # Only style transfer
```

### Custom Run (Interactive)
```bash
# Request interactive session (for debugging)
srun --partition=UGGPU-TC1 --qos=normal --gres=gpu:1 --mem=32G --time=60 --pty bash

# Then run manually
source activate diffusion_exp
python run_all_experiments.py --image test.jpg --output-dir ./outputs --experiment img2img
```

## Job Management

```bash
# Check job queue
squeue -u $USER

# View job details
scontrol show jobid <JOB_ID>

# Cancel job
scancel <JOB_ID>

# View output logs
tail -f logs/output_diffusion_exp_*.out
tail -f logs/error_diffusion_exp_*.err

# Check resource usage (after job completes)
seff <JOB_ID>
```

## Output Structure

After experiments complete:

```
experiments/
└── outputs/
    ├── img2img_20240115_143022/
    │   ├── metadata.json
    │   ├── pix2pix_landscape_make_it_winter.png
    │   ├── pix2pix_landscape_make_it_sunset.png
    │   └── sd_img2img_landscape_sunset_s75.png
    │
    ├── inpainting_20240115_144530/
    │   ├── metadata.json
    │   └── sd-inpainting_photo1_a_beautiful_garden.png
    │
    ├── style_20240115_150012/
    │   ├── metadata.json
    │   ├── edges_landscape.png
    │   ├── style_img2img_landscape_anime_s50.png
    │   └── style_controlnet_landscape_anime.png
    │
    └── experiment_summary_20240115_153045.json
```

## Resource Limits

The HPC has the following limits per user (normal QoS):

| Resource | Limit |
|----------|-------|
| GPU | 1 card (V100 32GB) |
| CPU | 20 cores max |
| Memory | 64GB max |
| Time | 6 hours max |
| Jobs | 2 concurrent |

The job script requests:
- 1 GPU, 48GB RAM, 8 CPUs, 6 hours

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or run fewer experiments
python run_all_experiments.py --experiment img2img --no-controlnet
```

### Job Timeout
- Split experiments: run img2img, inpainting, style separately
- Each type takes ~1-2 hours with 3-5 images

### CUDA Not Available
```bash
# Check module is loaded
module list
module load cuda/12.4

# Verify in Python
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Download Slow
- First run downloads models (~10GB total)
- Subsequent runs use cache
- Cache location: `~/.cache/huggingface`

## File Transfer

### Upload Images (from local machine)
```bash
# Using scp
scp -r ./test_images username@10.96.189.11:~/school_work/fyp/experiments/

# Using rsync (better for large files)
rsync -avz --progress ./test_images username@10.96.189.11:~/school_work/fyp/experiments/
```

### Download Results
```bash
scp -r username@10.96.189.11:~/school_work/fyp/experiments/outputs ./
```

## Support

- HPC Support: CCDSgpu-tc@ntu.edu.sg
- Check cluster status: `sinfo`
- View your quota: `ncdu ~`
