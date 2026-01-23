"""
Image restoration API endpoints.

Handles:
- Face enhancement (CodeFormer, GFPGAN)
- Image upscaling (Real-ESRGAN)
- Scratch and artifact removal
- B&W image colorization
"""

import time
from fastapi import APIRouter

from app.schemas.image import RestorationRequest, ImageResponse
from app.services.restoration import get_restoration_service

router = APIRouter()


@router.post("/", response_model=ImageResponse)
async def restore_image(request: RestorationRequest) -> ImageResponse:
    """
    Restore an old or damaged image.

    - **image**: Base64 encoded input image
    - **enable_face_enhance**: Enable face enhancement
    - **face_model**: 'codeformer' or 'gfpgan'
    - **fidelity**: CodeFormer fidelity (0=quality, 1=fidelity)
    - **upscale**: Upscaling factor ('none', '2x', '4x')
    - **enable_scratch_removal**: Enable scratch removal
    - **enable_colorize**: Enable colorization for B&W
    """
    start_time = time.time()

    try:
        service = get_restoration_service()
        result_b64, error, processing_time = service.restore(
            image_b64=request.image,
            enable_face_enhance=request.enable_face_enhance,
            face_model=request.face_model.value,
            fidelity=request.fidelity,
            upscale=request.upscale.value,
            enable_scratch_removal=request.enable_scratch_removal,
            enable_colorize=request.enable_colorize,
        )

        if error:
            return ImageResponse(
                success=False,
                error=error,
                processing_time=processing_time,
            )

        # Build model_used string
        models_used = []
        if request.enable_face_enhance:
            models_used.append(request.face_model.value)
        if request.upscale.value != "none":
            models_used.append(f"real-esrgan-{request.upscale.value}")
        if request.enable_scratch_removal:
            models_used.append("denoising")
        if request.enable_colorize:
            models_used.append("colorization")

        return ImageResponse(
            success=True,
            image=result_b64,
            model_used="+".join(models_used) if models_used else "none",
            processing_time=processing_time,
        )

    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time,
        )


@router.get("/models")
async def list_restoration_models():
    """List available restoration models and their capabilities."""
    return {
        "face_enhancement": [
            {
                "key": "codeformer",
                "name": "CodeFormer",
                "description": "Best quality face restoration with adjustable fidelity",
                "vram": "2-4 GB",
            },
            {
                "key": "gfpgan",
                "name": "GFPGAN",
                "description": "Fast face restoration, good for most cases",
                "vram": "2-4 GB",
            },
        ],
        "upscaling": [
            {
                "key": "real-esrgan-2x",
                "name": "Real-ESRGAN 2x",
                "description": "2x upscaling with artifact removal",
                "vram": "2-4 GB",
            },
            {
                "key": "real-esrgan-4x",
                "name": "Real-ESRGAN 4x",
                "description": "4x upscaling with artifact removal",
                "vram": "4-6 GB",
            },
        ],
        "other": [
            {
                "key": "scratch-removal",
                "name": "Scratch Removal",
                "description": "Remove scratches and artifacts from old photos",
            },
            {
                "key": "colorization",
                "name": "Colorization",
                "description": "Colorize black and white photos",
            },
        ],
    }


@router.get("/options")
async def get_restoration_options():
    """Get available restoration options and their defaults."""
    return {
        "face_models": ["codeformer", "gfpgan"],
        "upscale_options": ["none", "2x", "4x"],
        "defaults": {
            "enable_face_enhance": True,
            "face_model": "codeformer",
            "fidelity": 0.5,
            "upscale": "2x",
            "enable_scratch_removal": False,
            "enable_colorize": False,
        },
        "fidelity_description": {
            "0.0": "Maximum quality enhancement (may alter face)",
            "0.5": "Balanced (recommended)",
            "1.0": "Maximum fidelity to original (minimal changes)",
        },
    }
