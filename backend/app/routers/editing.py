"""
Image editing API endpoints.
Handles instruction-based editing, background replacement, and modifications.
"""

import time
from fastapi import APIRouter, HTTPException

from app.schemas.image import EditingRequest, ImageResponse
from app.services.huggingface import get_hf_service

router = APIRouter()


@router.post("/", response_model=ImageResponse)
async def edit_image(request: EditingRequest) -> ImageResponse:
    """
    Edit an image using natural language instructions.

    - **image**: Base64 encoded input image
    - **instruction**: Natural language edit instruction (e.g., "make it winter", "add sunglasses")
    - **mode**: 'instruct' for InstructPix2Pix, 'img2img' for standard transformation
    - **strength**: How much to transform (0.0-1.0)
    """
    start_time = time.time()

    try:
        service = get_hf_service()
        result_b64 = await service.edit_image(
            image_b64=request.image,
            instruction=request.instruction,
            mode=request.mode.value,
            guidance_scale=request.guidance_scale,
            image_guidance_scale=request.image_guidance_scale,
            strength=request.strength,
        )

        processing_time = time.time() - start_time

        return ImageResponse(
            success=True,
            image=result_b64,
            model_used=f"editing-{request.mode.value}",
            processing_time=processing_time,
        )

    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time,
        )


@router.get("/examples")
async def get_editing_examples():
    """Get example editing instructions."""
    return {
        "examples": [
            {"instruction": "make it winter with snow", "category": "weather"},
            {"instruction": "make it sunset", "category": "lighting"},
            {"instruction": "add sunglasses", "category": "add_object"},
            {"instruction": "remove the background", "category": "removal"},
            {"instruction": "turn into a painting", "category": "style"},
            {"instruction": "make it look vintage", "category": "effect"},
            {"instruction": "add rain and clouds", "category": "weather"},
            {"instruction": "change hair color to blonde", "category": "modification"},
        ]
    }
