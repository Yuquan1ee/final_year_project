"""
Inpainting API endpoints.
Handles object removal, region filling, and context-aware inpainting.
"""

from fastapi import APIRouter

from app.schemas.image import InpaintingRequest, ImageResponse
from app.services.diffusion import get_diffusion_service

router = APIRouter()


@router.post("/", response_model=ImageResponse)
async def inpaint_image(request: InpaintingRequest) -> ImageResponse:
    """
    Perform inpainting on an image.

    - **image**: Base64 encoded input image
    - **mask**: Base64 encoded mask (white areas will be inpainted)
    - **prompt**: Description of what to generate in the masked region
    - **model**: Model to use (sd-inpainting, sdxl-inpainting, kandinsky-inpainting, flux-fill)
    """
    service = get_diffusion_service()
    result_b64, error, processing_time = service.inpaint(
        image_b64=request.image,
        mask_b64=request.mask,
        prompt=request.prompt,
        model_key=request.model.value,
        negative_prompt=request.negative_prompt,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
    )

    if error:
        return ImageResponse(
            success=False,
            error=error,
            model_used=request.model.value,
            processing_time=processing_time,
        )

    return ImageResponse(
        success=True,
        image=result_b64,
        model_used=request.model.value,
        processing_time=processing_time,
    )


@router.get("/models")
async def list_inpainting_models():
    """List available inpainting models."""
    return {
        "models": [
            {
                "key": "sd-inpainting",
                "name": "Stable Diffusion Inpainting",
                "description": "Standard SD 1.5 inpainting, fast and reliable",
                "vram": "5-7 GB",
            },
            {
                "key": "sdxl-inpainting",
                "name": "SDXL Inpainting",
                "description": "Higher quality SDXL-based inpainting",
                "vram": "10-12 GB",
            },
            {
                "key": "kandinsky-inpainting",
                "name": "Kandinsky 2.2 Inpainting",
                "description": "Alternative model with different aesthetic",
                "vram": "6-8 GB",
            },
            {
                "key": "flux-fill",
                "name": "FLUX.1 Fill",
                "description": "State-of-the-art inpainting from Black Forest Labs",
                "vram": "22-24 GB",
            },
        ]
    }
