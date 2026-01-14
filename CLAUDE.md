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

This project focuses on leveraging diffusion models (e.g., Stable Diffusion, Imagen, or DALLE-2) for image editing, inpainting, and style transfer. The model will enable context-aware modifications, such as object removal, background replacement, and style transformation, by learning to reconstruct and modify images while preserving realism. Applications include photo restoration, creative design, and AI-assisted content generation.

### Literature Review Papers
- Denoising Diffusion Probabilistic Models (DDPM)
- Denoising Diffusion Implicit Models (DDIM)
- Latent Diffusion Models (Stable Diffusion)
- ControlNet (conditional control for diffusion models)
- LoRA (Low-Rank Adaptation for efficient fine-tuning)

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
├── backend/                    # FastAPI backend server
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── routers/           # API endpoints (inpainting, editing, style)
│   │   ├── services/          # HuggingFace API integration
│   │   └── schemas/           # Pydantic models
│   └── requirements.txt
├── frontend/                   # React frontend (Vite + TypeScript)
│   └── src/
│       ├── components/        # UI components
│       ├── services/          # API client
│       └── types/             # TypeScript types
├── experiments/               # Model testing scripts
│   ├── test_inpainting.py
│   ├── test_img2img.py
│   └── test_style_transfer.py
├── literature_review/         # Reference papers
└── reference_fyp_report/      # Reference FYP reports
```

## Development Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your HF_API_TOKEN
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Testing Models
```bash
cd experiments
pip install -r requirements.txt
cp .env.example .env  # Add your HF_API_TOKEN
python test_img2img.py --image photo.jpg --prompt "make it winter"
```

## Tech Stack

- **Backend:** FastAPI, HuggingFace Inference API
- **Frontend:** React, TypeScript, Vite
- **Models:** Stable Diffusion, InstructPix2Pix, ControlNet
