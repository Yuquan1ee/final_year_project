"""
Style transfer API endpoints.
Handles artistic style transfer, preset styles, and custom style prompts.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.image import StyleTransferRequest, ImageResponse
from app.services.diffusion import get_diffusion_service, STYLE_PROMPTS

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
    service = get_diffusion_service()

    # Map mode to model key
    model_key = "sdxl-img2img"  # default
    if request.mode.value == "img2img":
        model_key = "sdxl-img2img"
    # TODO: Add controlnet and ip-adapter modes

    result_b64, error, processing_time = service.style_transfer(
        image_b64=request.image,
        style=request.style,
        model_key=model_key,
        strength=request.strength,
    )

    if error:
        return ImageResponse(
            success=False,
            error=error,
            model_used=f"style-{request.mode.value}",
            processing_time=processing_time,
        )

    return ImageResponse(
        success=True,
        image=result_b64,
        model_used=f"style-{request.mode.value}",
        processing_time=processing_time,
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
