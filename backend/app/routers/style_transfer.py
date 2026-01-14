"""
Style transfer API endpoints.
Handles artistic style transfer, preset styles, and custom style prompts.
"""

import time
from fastapi import APIRouter, HTTPException

from app.schemas.image import StyleTransferRequest, ImageResponse
from app.services.huggingface import get_hf_service, STYLE_PROMPTS

router = APIRouter()


@router.post("/", response_model=ImageResponse)
async def apply_style(request: StyleTransferRequest) -> ImageResponse:
    """
    Apply style transfer to an image.

    - **image**: Base64 encoded content image
    - **style**: Style preset name or custom style prompt
    - **strength**: How much to stylize (0.0-1.0)
    - **mode**: 'img2img', 'controlnet', or 'ip-adapter'
    """
    start_time = time.time()

    try:
        service = get_hf_service()
        result_b64 = await service.style_transfer(
            image_b64=request.image,
            style=request.style,
            strength=request.strength,
        )

        processing_time = time.time() - start_time

        return ImageResponse(
            success=True,
            image=result_b64,
            model_used=f"style-{request.mode.value}",
            processing_time=processing_time,
        )

    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time,
        )


@router.get("/presets")
async def list_style_presets():
    """List available style presets."""
    presets = []
    for key, prompt in STYLE_PROMPTS.items():
        presets.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "prompt": prompt,
        })
    return {"presets": presets}


@router.get("/presets/{preset_key}")
async def get_style_preset(preset_key: str):
    """Get details of a specific style preset."""
    if preset_key not in STYLE_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Style preset '{preset_key}' not found")

    return {
        "key": preset_key,
        "name": preset_key.replace("_", " ").title(),
        "prompt": STYLE_PROMPTS[preset_key],
    }
