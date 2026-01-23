# Diffusion Models for Intelligent Image Editing and Inpainting

FYP Project (CCDS25-0278) - NTU AY 2025/2026

## Overview

This project leverages state-of-the-art diffusion models for intelligent image editing. The web application provides three main features powered by AI:

- **Inpainting**: Remove unwanted objects or fill regions intelligently
- **Style Transfer**: Apply artistic styles (anime, oil painting, watercolor, etc.)
- **Restoration**: Restore old or damaged photos, enhance faces, upscale images

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU with 8GB+ VRAM (for backend)

### Backend (requires GPU)

The backend runs diffusion models locally using PyTorch and Diffusers. It's designed to run on cloud GPU instances (Google Colab, AWS, etc.).

```bash
cd backend
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
# API runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install

# Configure backend URL (optional, defaults to localhost:8000)
echo "VITE_API_URL=http://localhost:8000" > .env

npm run dev
# UI runs at http://localhost:5173
```

### For Google Colab Deployment

1. Run the backend on Colab with GPU runtime
2. Use ngrok to expose the server: `ngrok http 8000`
3. Update frontend `.env` with the ngrok URL
4. Rebuild frontend: `npm run build`

## Features

### Inpainting
- Upload image and draw mask over areas to edit
- Enter a prompt describing what to generate
- Supports: SD Inpainting, SDXL Inpainting, Kandinsky, FLUX.1 Fill

### Style Transfer
- Upload image and select a style preset or enter custom description
- Adjust style strength (30-90%)
- Presets: Anime, Oil Painting, Watercolor, Sketch, Cyberpunk, Ghibli, etc.

### Restoration
- Upload old/damaged photo
- Options: Face enhancement (CodeFormer/GFPGAN), Upscaling (2x/4x), Scratch removal
- Automatic processing based on selected options

## Project Structure

```
.
├── backend/              # FastAPI backend with GPU inference
│   ├── app/
│   │   ├── main.py       # FastAPI app
│   │   ├── routers/      # API endpoints
│   │   ├── services/     # ML model inference
│   │   └── schemas/      # Pydantic models
│   ├── requirements.txt
│   └── README.md
├── frontend/             # React + TypeScript + Tailwind
│   ├── src/
│   │   ├── api/          # API client
│   │   ├── components/   # Tab components
│   │   └── App.tsx
│   ├── package.json
│   └── README.md
├── literature_review/    # Research papers
├── reference_fyp_report/ # Reference reports
├── CLAUDE.md            # Development guide
└── README.md            # This file
```

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, PyTorch, Diffusers, Transformers |
| Inpainting | SD Inpainting, SDXL Inpainting, Kandinsky, FLUX.1 Fill |
| Style Transfer | SDXL img2img |
| Restoration | CodeFormer, GFPGAN, Real-ESRGAN |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inpainting/` | POST | Inpaint masked regions |
| `/api/inpainting/models` | GET | List available inpainting models |
| `/api/style/` | POST | Apply style transfer |
| `/api/style/presets` | GET | List style presets |
| `/api/restoration/` | POST | Restore/enhance images |
| `/api/restoration/models` | GET | List restoration models |
| `/api/health` | GET | Health check with GPU info |

## VRAM Requirements

| Model | VRAM (FP16) | Notes |
|-------|-------------|-------|
| SD Inpainting | 5-7 GB | Fast, good quality |
| SDXL Inpainting | 10-12 GB | Higher quality |
| Kandinsky Inpainting | 6-8 GB | Alternative aesthetic |
| FLUX.1 Fill | 22-24 GB | State-of-the-art |
| CodeFormer | 2-4 GB | Face restoration |
| GFPGAN | 2-4 GB | Face restoration |
| Real-ESRGAN | 2-6 GB | Upscaling |

## Model Selection Guide

| Task | Recommended Model | Why |
|------|-------------------|-----|
| General Inpainting | SD Inpainting | Good balance of speed and quality |
| High Quality Inpainting | SDXL Inpainting | Better details |
| Best Quality (if VRAM allows) | FLUX.1 Fill | Current SOTA |
| Face Restoration | CodeFormer | Best for old/damaged photos |
| Super Resolution | Real-ESRGAN | Best general upscaling |

## References

### Inpainting
- [FLUX.1 Fill Documentation](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)

### Restoration
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## Project Info

| Field | Value |
|-------|-------|
| Project # | CCDS25-0278 |
| Student | Lee Yu Quan |
| Supervisor | Prof Zhang Hanwang |
| Lab | Multimedia and Interacting Computing Lab (MICL) |
| Institution | Nanyang Technological University |

## License

MIT License
