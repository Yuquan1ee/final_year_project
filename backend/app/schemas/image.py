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
    # Standard models
    SD_INPAINTING = "sd-inpainting"
    KANDINSKY = "kandinsky-inpainting"
    # SDXL variants
    SDXL_INPAINTING = "sdxl-inpainting"
    SDXL_INPAINTING_8BIT = "sdxl-inpainting-8bit"
    SDXL_INPAINTING_4BIT = "sdxl-inpainting-4bit"
    # FLUX.1 Fill variants
    FLUX_FILL = "flux-fill"
    FLUX_FILL_FP16 = "flux-fill-fp16"
    FLUX_FILL_8BIT = "flux-fill-8bit"
    FLUX_FILL_4BIT = "flux-fill-4bit"
    FLUX_FILL_NF4 = "flux-fill-nf4"


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


class StyleModel(str, Enum):
    """Available style transfer models."""
    # SDXL variants
    SDXL_IMG2IMG = "sdxl-img2img"
    SDXL_IMG2IMG_8BIT = "sdxl-img2img-8bit"
    SDXL_IMG2IMG_4BIT = "sdxl-img2img-4bit"
    # SD 1.5
    SD_IMG2IMG = "sd-img2img"


class StyleTransferRequest(BaseModel):
    """Request for style transfer operation."""
    image: str = Field(..., description="Base64 encoded content image")
    style: str = Field(
        ...,
        description="Style preset name or custom style prompt"
    )
    model: StyleModel = Field(
        default=StyleModel.SDXL_IMG2IMG,
        description="Model to use for style transfer"
    )
    strength: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Style strength (higher = more stylized)"
    )


# =============================================================================
# Restoration Schemas
# =============================================================================


class FaceModel(str, Enum):
    """Available face enhancement models."""
    CODEFORMER = "codeformer"
    GFPGAN = "gfpgan"


class UpscaleOption(str, Enum):
    """Upscaling options."""
    NONE = "none"
    UPSCALE_2X = "2x"
    UPSCALE_4X = "4x"


class RestorationRequest(BaseModel):
    """Request for image restoration operation."""
    image: str = Field(..., description="Base64 encoded input image")
    enable_face_enhance: bool = Field(
        default=True,
        description="Enable face enhancement (CodeFormer/GFPGAN)"
    )
    face_model: FaceModel = Field(
        default=FaceModel.CODEFORMER,
        description="Face enhancement model to use"
    )
    fidelity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="CodeFormer fidelity (0=quality, 1=fidelity to input)"
    )
    upscale: UpscaleOption = Field(
        default=UpscaleOption.UPSCALE_2X,
        description="Upscaling factor (none, 2x, 4x)"
    )
    enable_scratch_removal: bool = Field(
        default=False,
        description="Enable scratch and artifact removal"
    )
    enable_colorize: bool = Field(
        default=False,
        description="Enable colorization for B&W images"
    )
