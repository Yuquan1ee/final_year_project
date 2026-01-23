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
│   │   ├── routers/           # API endpoints (inpainting, style, restoration)
│   │   ├── services/          # HuggingFace API integration
│   │   └── schemas/           # Pydantic models
│   └── requirements.txt
├── frontend/                   # React frontend (Vite + TypeScript + Tailwind)
│   ├── src/
│   │   ├── components/        # Tab components
│   │   │   ├── HomeTab.tsx    # Project info and welcome page
│   │   │   ├── InpaintingTab.tsx    # Inpainting feature
│   │   │   ├── StyleTransferTab.tsx # Style transfer feature
│   │   │   └── RestorationTab.tsx   # Photo restoration feature
│   │   ├── App.tsx            # Main app with tab navigation
│   │   ├── main.tsx           # React entry point
│   │   └── index.css          # Tailwind CSS import
│   ├── index.html             # HTML entry point
│   ├── package.json           # Dependencies
│   ├── vite.config.ts         # Vite + Tailwind configuration
│   └── tsconfig.json          # TypeScript configuration
├── hpc_instructions/          # NTU HPC cluster documentation
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
# API runs at http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

### Build Frontend for Production
```bash
cd frontend
npm run build    # Outputs to dist/
npm run preview  # Preview production build
```

## Tech Stack

- **Backend:** FastAPI, HuggingFace Inference API
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS
- **Models:** FLUX.1 Fill, Stable Diffusion, ControlNet, IP-Adapter, CodeFormer, Real-ESRGAN

## Frontend Architecture

The frontend uses a tab-based navigation with four main sections:

| Tab | Component | Purpose |
|-----|-----------|---------|
| Home | `HomeTab.tsx` | Project info, feature overview, diffusion model explanation |
| Inpainting | `InpaintingTab.tsx` | Remove objects or fill regions with AI |
| Style Transfer | `StyleTransferTab.tsx` | Apply artistic styles to images |
| Restoration | `RestorationTab.tsx` | Restore and enhance old/damaged photos |
