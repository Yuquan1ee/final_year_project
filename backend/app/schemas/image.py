"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ImageRequest(BaseModel):
    """Base request with an image."""
    image: str = Field(..., description="Base64 encoded image")


class ImageResponse(BaseModel):
    """Response containing a generated image."""
    success: bool
    image: Optional[str] = Field(None, description="Base64 encoded result image")
    error: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None


class InpaintingModel(str, Enum):
    """Available inpainting models."""
    SD_INPAINTING = "sd-inpainting"
    SDXL_INPAINTING = "sdxl-inpainting"
    KANDINSKY = "kandinsky-inpainting"


class InpaintingRequest(BaseModel):
    """Request for inpainting operation."""
    image: str = Field(..., description="Base64 encoded input image")
    mask: str = Field(..., description="Base64 encoded mask (white=inpaint, black=keep)")
    prompt: str = Field(..., description="Description of what to generate in masked area")
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, deformed",
        description="What to avoid in generation"
    )
    model: InpaintingModel = Field(
        default=InpaintingModel.SD_INPAINTING,
        description="Model to use for inpainting"
    )
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=30, ge=10, le=100)


class EditingMode(str, Enum):
    """Available editing modes."""
    INSTRUCT = "instruct"  # InstructPix2Pix style
    IMG2IMG = "img2img"    # Standard img2img


class EditingRequest(BaseModel):
    """Request for image editing operation."""
    image: str = Field(..., description="Base64 encoded input image")
    instruction: str = Field(..., description="Editing instruction (e.g., 'make it winter')")
    mode: EditingMode = Field(default=EditingMode.INSTRUCT)
    strength: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Edit strength (0=no change, 1=complete transformation)"
    )
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    image_guidance_scale: float = Field(
        default=1.5, ge=0.0, le=5.0,
        description="How much to follow original image (InstructPix2Pix only)"
    )


class StylePreset(str, Enum):
    """Predefined style presets."""
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    ANIME = "anime"
    SKETCH = "sketch"
    CYBERPUNK = "cyberpunk"
    GHIBLI = "ghibli"
    PIXEL_ART = "pixel_art"
    POP_ART = "pop_art"
    IMPRESSIONIST = "impressionist"
    CUSTOM = "custom"


class StyleTransferMode(str, Enum):
    """Style transfer methods."""
    IMG2IMG = "img2img"
    CONTROLNET = "controlnet"
    IP_ADAPTER = "ip-adapter"


class StyleTransferRequest(BaseModel):
    """Request for style transfer operation."""
    image: str = Field(..., description="Base64 encoded content image")
    style: str = Field(
        ...,
        description="Style preset name or custom style prompt"
    )
    style_image: Optional[str] = Field(
        None,
        description="Base64 encoded style reference image (for IP-Adapter mode)"
    )
    mode: StyleTransferMode = Field(default=StyleTransferMode.IMG2IMG)
    strength: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Style strength (higher = more stylized)"
    )
    preserve_structure: bool = Field(
        default=True,
        description="Use ControlNet to preserve image structure"
    )
