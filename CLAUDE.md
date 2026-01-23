# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Title:** Diffusion Models for Intelligent Image Editing and Inpainting
**Project #:** CCDS25-0278
**Student:** Lee Yu Quan
**Supervisor:** Prof Zhang Hanwang
**Lab:** Multimedia and Interacting Computing Lab (MICL)
**Area:** Artificial Intelligence

### Project Summary

This project focuses on leveraging diffusion models (e.g., Stable Diffusion, FLUX.1) for image editing, inpainting, and style transfer. The web application provides three main features:

1. **Inpainting** - Remove objects or fill regions using AI (SD, SDXL, Kandinsky, FLUX.1 Fill)
2. **Style Transfer** - Apply artistic styles (anime, oil painting, watercolor, etc.)
3. **Restoration** - Enhance faces (CodeFormer/GFPGAN), upscale (Real-ESRGAN), remove scratches

### Literature Review Papers
- Denoising Diffusion Probabilistic Models (DDPM)
- Denoising Diffusion Implicit Models (DDIM)
- Latent Diffusion Models (Stable Diffusion)
- ControlNet (conditional control for diffusion models)
- LoRA (Low-Rank Adaptation for efficient fine-tuning)

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Frontend UI | ✅ Complete | React + TypeScript + Tailwind |
| Frontend ↔ Backend API | ✅ Connected | All 3 tabs connected |
| Backend API Structure | ✅ Complete | FastAPI with 3 endpoints |
| Inpainting Service | ✅ Ready | Needs GPU to run |
| Style Transfer Service | ✅ Ready | Needs GPU to run |
| Restoration Service | ✅ Ready | CodeFormer, GFPGAN, Real-ESRGAN |
| Colorization | ⚠️ Placeholder | Not yet implemented |

## Next Task

**Test the backend on Google Colab:**
1. Create a Colab notebook that clones the repo and installs dependencies
2. Use ngrok to expose the FastAPI server publicly
3. Configure the frontend to point to the ngrok URL
4. Test all three features (inpainting, style transfer, restoration)
5. Document any issues or required fixes

## FYP Timeline (AY 2025/2026)

| Date | Milestone |
|------|-----------|
| 11 Aug 2025 (Sem1Wk1) | FYP officially starts |
| 1 Sep 2025 (Sem1Wk4) | Submission of Project Plan/Strategy to Supervisor |
| 26 Jan 2026 (Sem2Wk3) | Submission of Interim Report |
| 23 Mar 2026 (Sem2Wk10) | Submission of Final Report |
| 17 Apr 2026 (Sem2Wk13) | Submission of Amended Final Report |
| 8-13 May 2026 | Oral Presentation (20 min + 10 min Q&A) |

## Repository Structure

```
.
├── backend/                    # FastAPI backend (runs on GPU)
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── routers/           # API endpoints
│   │   │   ├── inpainting.py  # /api/inpainting/
│   │   │   ├── style_transfer.py  # /api/style/
│   │   │   └── restoration.py # /api/restoration/
│   │   ├── services/          # ML model inference
│   │   │   ├── diffusion.py   # Diffusers-based inference
│   │   │   └── restoration.py # CodeFormer, GFPGAN, Real-ESRGAN
│   │   └── schemas/           # Pydantic request/response models
│   ├── requirements.txt
│   └── README.md
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── api/               # API client for backend
│   │   │   ├── config.ts      # API URL configuration
│   │   │   ├── imageApi.ts    # API functions
│   │   │   └── index.ts
│   │   ├── components/        # Tab components
│   │   │   ├── HomeTab.tsx
│   │   │   ├── InpaintingTab.tsx
│   │   │   ├── StyleTransferTab.tsx
│   │   │   └── RestorationTab.tsx
│   │   ├── App.tsx            # Main app with tab navigation
│   │   └── main.tsx
│   ├── .env.example           # Environment variable template
│   ├── package.json
│   └── README.md
├── hpc_instructions/          # NTU HPC cluster documentation
├── literature_review/         # Reference papers
├── reference_fyp_report/      # Reference FYP reports
├── CLAUDE.md                  # This file
└── README.md                  # Project overview
```

## Development Commands

### Backend (requires GPU)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# API runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

### Configure Frontend API URL
```bash
# Create frontend/.env file
echo "VITE_API_URL=http://localhost:8000" > frontend/.env

# For Colab, use ngrok URL:
echo "VITE_API_URL=https://xxxx.ngrok-free.app" > frontend/.env
```

### Build Frontend for Production
```bash
cd frontend
npm run build    # Outputs to dist/
npm run preview  # Preview production build
```

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, PyTorch, Diffusers, Transformers |
| Inpainting | SD Inpainting, SDXL Inpainting, Kandinsky, FLUX.1 Fill |
| Style Transfer | SDXL img2img with style prompts |
| Restoration | CodeFormer, GFPGAN, Real-ESRGAN |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inpainting/` | POST | Inpaint masked regions |
| `/api/inpainting/models` | GET | List available models |
| `/api/style/` | POST | Apply style transfer |
| `/api/style/presets` | GET | List style presets |
| `/api/restoration/` | POST | Restore/enhance images |
| `/api/restoration/models` | GET | List restoration models |
| `/api/health` | GET | Health check with GPU info |

## VRAM Requirements

| Model | VRAM (FP16) |
|-------|-------------|
| SD Inpainting | 5-7 GB |
| SDXL Inpainting | 10-12 GB |
| Kandinsky Inpainting | 6-8 GB |
| FLUX.1 Fill | 22-24 GB |
| CodeFormer | 2-4 GB |
| GFPGAN | 2-4 GB |
| Real-ESRGAN | 2-6 GB |

## Key Files to Know

| File | Purpose |
|------|---------|
| `backend/app/services/diffusion.py` | Main diffusion model inference |
| `backend/app/services/restoration.py` | Face/upscale restoration |
| `frontend/src/api/imageApi.ts` | Frontend API client |
| `frontend/src/api/config.ts` | API URL configuration |
