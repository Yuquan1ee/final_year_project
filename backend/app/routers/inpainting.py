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
            # Standard models
            {
                "key": "sd-inpainting",
                "name": "Stable Diffusion Inpainting",
                "description": "Standard SD 1.5 inpainting, fast and reliable",
                "vram": "5-7 GB",
            },
            {
                "key": "kandinsky-inpainting",
                "name": "Kandinsky 2.2 Inpainting",
                "description": "Alternative model with different aesthetic",
                "vram": "6-8 GB",
            },
            # SDXL variants
            {
                "key": "sdxl-inpainting",
                "name": "SDXL Inpainting",
                "description": "Higher quality SDXL-based inpainting",
                "vram": "10-12 GB",
            },
            {
                "key": "sdxl-inpainting-8bit",
                "name": "SDXL Inpainting (8-bit)",
                "description": "SDXL inpainting with 8-bit quantization",
                "vram": "~6 GB",
            },
            {
                "key": "sdxl-inpainting-4bit",
                "name": "SDXL Inpainting (4-bit)",
                "description": "SDXL inpainting with 4-bit quantization",
                "vram": "~4 GB",
            },
            # FLUX.1 Fill variants
            {
                "key": "flux-fill",
                "name": "FLUX.1 Fill (Full)",
                "description": "State-of-the-art inpainting, full precision",
                "vram": "22-24 GB",
            },
            {
                "key": "flux-fill-8bit",
                "name": "FLUX.1 Fill (8-bit)",
                "description": "FLUX.1 Fill with 8-bit quantization",
                "vram": "~16 GB",
            },
            {
                "key": "flux-fill-4bit",
                "name": "FLUX.1 Fill (4-bit)",
                "description": "FLUX.1 Fill with 4-bit quantization",
                "vram": "~10 GB",
            },
            {
                "key": "flux-fill-nf4",
                "name": "FLUX.1 Fill (NF4)",
                "description": "FLUX.1 Fill with NF4 quantization (recommended)",
                "vram": "~10 GB",
            },
        ]
    }
