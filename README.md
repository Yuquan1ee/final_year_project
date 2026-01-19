# Diffusion Models for Intelligent Image Editing and Inpainting

FYP Project (CCDS25-0278) - NTU AY 2025/2026

## Features

- **Image Editing**: Natural language instructions to modify images (e.g., "make it winter", "add sunglasses")
- **Inpainting**: Remove objects or fill regions intelligently
- **Style Transfer**: Apply artistic styles (anime, oil painting, watercolor, etc.)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- HuggingFace API Token ([Get one here](https://huggingface.co/settings/tokens))

### Run the Full Application

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your HF_API_TOKEN

uvicorn app.main:app --reload
# API runs at http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

## Project Structure

```
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── experiments/      # Model testing scripts
├── literature_review/ # Research papers
└── reference_fyp_report/ # Reference reports
```

## API Endpoints

- `POST /api/editing/` - Edit image with natural language
- `POST /api/inpainting/` - Inpaint masked regions
- `POST /api/style/` - Apply style transfer
- `GET /api/health` - Health check

## Tech Stack

- FastAPI + HuggingFace Inference API
- React + TypeScript + Vite
- Stable Diffusion, InstructPix2Pix, ControlNet


## Running Experiments on HPC

For VPN connection, SSH access, file transfer, and running experiments on the NTU HPC cluster, see [experiments/README_HPC.md](experiments/README_HPC.md).
