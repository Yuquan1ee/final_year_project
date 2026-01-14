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

### 1. Test Models First (Recommended)

```bash
# Set up experiments environment
cd experiments
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your HF_API_TOKEN

# Test image editing
python test_img2img.py --image your_photo.jpg --prompt "make it sunset"

# Test inpainting
python test_inpainting.py --image photo.jpg --create-mask  # Create a mask first
python test_inpainting.py --image photo.jpg --mask photo_mask.png --prompt "a cat"

# Test style transfer
python test_style_transfer.py --image photo.jpg --style anime
```

### 2. Run the Full Application

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
