# Diffusion Image Editor - Backend

FastAPI backend for intelligent image editing using diffusion models. Designed to run on cloud GPU instances (Google Colab, AWS, etc.).

## Features

- **Inpainting**: Remove objects or fill regions using SD, SDXL, Kandinsky, or FLUX.1 Fill
- **Image Editing**: Edit images with natural language instructions (InstructPix2Pix)
- **Style Transfer**: Apply artistic styles (anime, oil painting, watercolor, etc.)
- **Restoration**: Enhance faces (CodeFormer/GFPGAN), upscale (Real-ESRGAN), remove scratches

## Requirements

- Python 3.10+
- CUDA-capable GPU with 8GB+ VRAM (16GB+ recommended for SDXL/FLUX models)
- ~20GB disk space for model weights

### VRAM Requirements by Model

| Model | VRAM (FP16) | Notes |
|-------|-------------|-------|
| SD Inpainting | 5-7 GB | Fast, good quality |
| SDXL Inpainting | 10-12 GB | Higher quality |
| FLUX.1 Fill | 22-24 GB | State-of-the-art |
| InstructPix2Pix | 6-8 GB | Text-based editing |
| CodeFormer | 2-4 GB | Face restoration |
| GFPGAN | 2-4 GB | Face restoration |
| Real-ESRGAN | 2-6 GB | Upscaling |

## Installation

### Local Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install CodeFormer (for face enhancement)
pip install codeformer-pip

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Google Colab

See the Colab notebook in the root directory, or run:

```python
# Install dependencies
!pip install fastapi uvicorn pyngrok diffusers transformers accelerate
!pip install gfpgan realesrgan basicsr facexlib

# Clone repo and start server
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO/backend

# Start with ngrok
from pyngrok import ngrok
import nest_asyncio
nest_asyncio.apply()

public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```
GET /api/health
```

Returns GPU status and loaded models.

### Inpainting

```
POST /api/inpainting/
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "mask": "base64_encoded_mask",
  "prompt": "a beautiful garden with flowers",
  "negative_prompt": "blurry, low quality",
  "model": "sd-inpainting",
  "guidance_scale": 7.5,
  "num_inference_steps": 30
}
```

**Available Models**: `sd-inpainting`, `sdxl-inpainting`, `kandinsky-inpainting`, `flux-fill`

```
GET /api/inpainting/models
```

### Image Editing

```
POST /api/editing/
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "instruction": "make it winter with snow",
  "mode": "instruct",
  "strength": 0.75,
  "guidance_scale": 7.5
}
```

**Modes**: `instruct` (InstructPix2Pix), `img2img` (Stable Diffusion)

```
GET /api/editing/examples
```

### Style Transfer

```
POST /api/style/
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "style": "anime",
  "strength": 0.6
}
```

**Style Presets**: `oil_painting`, `watercolor`, `anime`, `sketch`, `cyberpunk`, `ghibli`, `pixel_art`, `pop_art`, `impressionist`

```
GET /api/style/presets
GET /api/style/presets/{preset_key}
```

### Restoration

```
POST /api/restoration/
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "enable_face_enhance": true,
  "face_model": "codeformer",
  "fidelity": 0.5,
  "upscale": "2x",
  "enable_scratch_removal": false,
  "enable_colorize": false
}
```

**Face Models**: `codeformer`, `gfpgan`
**Upscale Options**: `none`, `2x`, `4x`

```
GET /api/restoration/models
GET /api/restoration/options
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, routers
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── inpainting.py    # /api/inpainting endpoints
│   │   ├── editing.py       # /api/editing endpoints
│   │   ├── style_transfer.py # /api/style endpoints
│   │   └── restoration.py   # /api/restoration endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── image.py         # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── diffusion.py     # Diffusion model inference
│       └── restoration.py   # Restoration model inference
├── requirements.txt
└── README.md
```

## Response Format

All image endpoints return:

```json
{
  "success": true,
  "image": "base64_encoded_result",
  "model_used": "sd-inpainting",
  "processing_time": 12.5,
  "error": null
}
```

On error:

```json
{
  "success": false,
  "image": null,
  "error": "Error message here",
  "processing_time": 0.5
}
```

## Environment Variables

Create a `.env` file (optional):

```bash
# Not required for local inference
# HF_API_TOKEN=your_token_here  # Only if using HF Inference API
```

## Development

### Running Tests

```bash
pytest tests/
```

### API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Out of Memory (OOM)

- Use smaller models (SD instead of SDXL)
- Reduce image size
- Enable attention slicing (already enabled by default)
- Use `fp16` precision (already enabled by default)

### Model Loading Slow

First request loads models into GPU memory (30-60s). Subsequent requests are faster as models stay cached.

### CUDA Not Available

Ensure PyTorch is installed with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## License

MIT License - See LICENSE file in root directory.
