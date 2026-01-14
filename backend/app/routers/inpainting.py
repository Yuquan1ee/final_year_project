"""
Inpainting API endpoints.
Handles object removal, region filling, and context-aware inpainting.
"""

import time
from fastapi import APIRouter, HTTPException

from app.schemas.image import InpaintingRequest, ImageResponse
from app.services.huggingface import get_hf_service

router = APIRouter()


@router.post("/", response_model=ImageResponse)
async def inpaint_image(request: InpaintingRequest) -> ImageResponse:
    """
    Perform inpainting on an image.

    - **image**: Base64 encoded input image
    - **mask**: Base64 encoded mask (white areas will be inpainted)
    - **prompt**: Description of what to generate in the masked region
    - **model**: Model to use (sd-inpainting, sdxl-inpainting, kandinsky-inpainting)
    """
    start_time = time.time()

    try:
        service = get_hf_service()
        result_b64 = await service.inpaint(
            image_b64=request.image,
            mask_b64=request.mask,
            prompt=request.prompt,
            model_key=request.model.value,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
        )

        processing_time = time.time() - start_time

        return ImageResponse(
            success=True,
            image=result_b64,
            model_used=request.model.value,
            processing_time=processing_time,
        )

    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time,
        )


@router.get("/models")
async def list_inpainting_models():
    """List available inpainting models."""
    return {
        "models": [
            {
                "key": "sd-inpainting",
                "name": "Stable Diffusion Inpainting",
                "description": "Standard SD inpainting model, fast and reliable",
            },
            {
                "key": "sdxl-inpainting",
                "name": "SDXL Inpainting",
                "description": "Higher quality, slower, better for detailed work",
            },
            {
                "key": "kandinsky-inpainting",
                "name": "Kandinsky 2.2 Inpainting",
                "description": "Alternative model with different aesthetic",
            },
        ]
    }
