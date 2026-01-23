# Diffusion Models for Intelligent Image Editing and Inpainting

FYP Project (CCDS25-0278) - NTU AY 2025/2026

## Overview

This project leverages state-of-the-art diffusion models for intelligent image editing. The web application provides three main features powered by AI:

- **Inpainting**: Remove unwanted objects or fill regions intelligently
- **Style Transfer**: Apply artistic styles (anime, oil painting, watercolor, etc.)
- **Restoration**: Restore old or damaged photos, enhance quality, remove artifacts

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- HuggingFace API Token ([Get one here](https://huggingface.co/settings/tokens))

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your HF_API_TOKEN

uvicorn app.main:app --reload
# API runs at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

## State-of-the-Art Open Source Models

### Inpainting Models

| Model | Source | HuggingFace | VRAM (FP16) | Description |
|-------|--------|-------------|-------------|-------------|
| **FLUX.1 Fill [dev]** ⭐ | Black Forest Labs | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | 22-24 GB | **Current SOTA** - 12B parameter official inpainting/outpainting model |
| **FLUX.1 Controlnet Inpainting** | Alimama Creative | [alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta) | 24+ GB | ControlNet-based inpainting for FLUX, 1024x1024 support |
| **BrushNet** | TencentARC (ECCV 2024) | [TencentARC/BrushNet](https://huggingface.co/TencentARC/BrushNet) | 6-8 GB | Plug-and-play inpainting with decomposed dual-branch diffusion |
| **Stable Diffusion XL Inpainting** | Stability AI | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | 10-12 GB | High-resolution inpainting based on SDXL |
| **Stable Diffusion Inpainting** | Stability AI | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) | 5-7 GB | Classic SD 1.5 based inpainting, fast inference |
| **Kandinsky 2.2 Inpainting** | Sber AI | [kandinsky-community/kandinsky-2-2-decoder-inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) | 6-8 GB | High-quality multilingual inpainting |
| **LaMa** | Samsung AI | [smartywu/big-lama](https://huggingface.co/smartywu/big-lama) | 2-4 GB | Large Mask Inpainting with Fourier Convolutions |

### Style Transfer Models

| Model | Source | HuggingFace | VRAM (FP16) | Description |
|-------|--------|-------------|-------------|-------------|
| **IP-Adapter** ⭐ | Tencent AI Lab | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) | 8-12 GB (SD1.5) | **Recommended** - Image prompt adapter for style/content transfer |
| **IP-Adapter FaceID** | Tencent AI Lab | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | 8-12 GB | Face-consistent style transfer with identity preservation |
| **FLUX.1 IP-Adapter** | InstantX | [InstantX/FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) | 24+ GB | IP-Adapter for FLUX models, 128 image tokens |
| **Stable Diffusion XL** | Stability AI | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 8-12 GB | Base model for img2img style transfer |
| **SDXL-Lightning** | ByteDance | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | 8-10 GB | Fast (<1s) style transfer at 1024x1024 |

### Control Models (ControlNet / ControlLoRA)

| Model | Source | HuggingFace | VRAM (FP16) | Description |
|-------|--------|-------------|-------------|-------------|
| **FLUX.1 Depth [dev]** | Black Forest Labs | [black-forest-labs/FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) | 22-24 GB | Depth-guided generation for FLUX |
| **FLUX.1 Canny [dev]** | Black Forest Labs | [black-forest-labs/FLUX.1-Canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) | 22-24 GB | Edge-guided generation for FLUX |
| **ControlNet (SD/SDXL)** | Lvmin Zhang | [lllyasviel/ControlNet](https://huggingface.co/lllyasviel/ControlNet) | 8-12 GB | Structure-preserving control (depth, canny, pose, etc.) |
| **ControlLoRA** | Community | [diffusers ControlLoRA](https://github.com/huggingface/diffusers) | 6-8 GB | Lightweight alternative to ControlNet (~7M params vs 4.7GB) |
| **InstantX ControlNet** | InstantX | [InstantX/FLUX.1-dev-Controlnet-Union](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union) | 24+ GB | Multi-condition ControlNet for FLUX |

### Restoration Models

| Model | Source | HuggingFace | VRAM (FP16) | Description |
|-------|--------|-------------|-------------|-------------|
| **CodeFormer** ⭐ | NTU S-Lab (NeurIPS 2022) | [sczhou/CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer) | 2-4 GB | **Recommended** - Blind face restoration with learned discrete codebook |
| **GFPGAN** | TencentARC | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | 2-4 GB | GAN-based face restoration built on StyleGAN2 |
| **Real-ESRGAN** ⭐ | Xintao Wang | [ai-forever/Real-ESRGAN](https://huggingface.co/ai-forever/Real-ESRGAN) | 2-6 GB* | **Recommended** - Real-world image super-resolution |
| **RestoreFormer++** | Microsoft | [sczhou/RestoreFormer](https://huggingface.co/sczhou/RestoreFormer) | 2-4 GB | Face restoration with multi-head cross-attention |
| **SwinIR** | ETH Zurich | [caidas/swinIR](https://huggingface.co/caidas/swinIR) | 2-4 GB | Image restoration using Swin Transformer |
| **Retinexformer** | ICCV 2023 | [papers/Retinexformer](https://huggingface.co/papers/2303.06705) | 2-4 GB | Low-light image enhancement |

*Real-ESRGAN VRAM scales with input image resolution. Use `--tile` option for large images.

### Model Selection Guide

| Task | Recommended Model | Why |
|------|-------------------|-----|
| **General Inpainting** | FLUX.1 Fill [dev] | Current SOTA, best quality |
| **Fast Inpainting** | SD Inpainting | Faster inference, good quality |
| **Large Area Removal** | LaMa | Specialized for large masks |
| **Style Transfer** | IP-Adapter + SDXL | Best style consistency |
| **Face Styling** | IP-Adapter FaceID | Preserves identity |
| **Structure Control** | FLUX Depth/Canny | Preserves composition |
| **Lightweight Control** | ControlLoRA | 600x smaller than ControlNet |
| **Face Restoration** | CodeFormer | Best for old/damaged photos |
| **Super Resolution** | Real-ESRGAN | Best general upscaling |
| **Low-light Enhancement** | Retinexformer | SOTA for dark images |

## Project Structure

```
.
├── backend/              # FastAPI backend
├── frontend/             # React + TypeScript + Tailwind CSS
│   └── src/
│       ├── components/   # Tab components (Home, Inpainting, StyleTransfer, Restoration)
│       └── App.tsx       # Main app with navigation
├── literature_review/    # Research papers
└── reference_fyp_report/ # Reference reports
```

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, HuggingFace Diffusers, PyTorch |
| AI Models | FLUX.1 Fill, IP-Adapter, ControlNet, CodeFormer, Real-ESRGAN |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inpainting/` | POST | Inpaint masked regions |
| `/api/style/` | POST | Apply style transfer |
| `/api/restoration/` | POST | Restore/enhance images |
| `/api/health` | GET | Health check |

## References

### Inpainting
- [FLUX.1 Fill Documentation](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [FLUX Tools Announcement](https://bfl.ai/announcements/24-11-21-tools)
- [BrushNet Paper (ECCV 2024)](https://huggingface.co/papers/2403.06976)
- [IOPaint - Open Source Inpainting Tool](https://github.com/Sanster/IOPaint)

### Style Transfer
- [IP-Adapter Diffusers Guide](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)
- [InstantStyle for Style Control](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter#style-and-layout-control)
- [Tencent AI Lab IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)

### Control Models
- [FLUX ControlNet Guide](https://flux-kontext.io/posts/flux-controlnet)
- [ControlLoRA GitHub](https://github.com/HighCWu/ControlLoRA)
- [FLUX Depth/Canny Models](https://huggingface.co/black-forest-labs)

### Restoration
- [CodeFormer Demo](https://huggingface.co/spaces/sczhou/CodeFormer)
- [Open Image Restoration Toolkit](https://github.com/titsitits/open-image-restoration)
- [Awesome Diffusion for Image Processing](https://github.com/lixinustc/Awesome-diffusion-model-for-image-processing)

## Project Info

| Field | Value |
|-------|-------|
| Project # | CCDS25-0278 |
| Student | Lee Yu Quan |
| Supervisor | Prof Zhang Hanwang |
| Lab | Multimedia and Interacting Computing Lab (MICL) |
| Institution | Nanyang Technological University |
