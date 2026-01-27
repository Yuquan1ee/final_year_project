"""
Local diffusion model service for GPU inference.

This service runs models locally using the diffusers library.
Designed to run on cloud GPU instances (Colab, AWS, etc.)
"""

import base64
import io
import time
from typing import Optional, Tuple, Dict, Any

import torch
from PIL import Image

# Lazy imports for model pipelines - loaded only when needed
# This prevents import errors when models aren't needed yet


# =============================================================================
# Model Configurations
# =============================================================================

INPAINTING_MODELS = {
    "sd-inpainting": {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "pipeline": "StableDiffusionInpaintPipeline",
        "torch_dtype": "float16",
    },
    "sdxl-inpainting": {
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pipeline": "StableDiffusionXLInpaintPipeline",
        "torch_dtype": "float16",
    },
    "kandinsky-inpainting": {
        "model_id": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        "pipeline": "KandinskyV22InpaintPipeline",
        "torch_dtype": "float16",
    },
    "flux-fill": {
        "model_id": "black-forest-labs/FLUX.1-Fill-dev",
        "pipeline": "FluxFillPipeline",
        "torch_dtype": "bfloat16",
    },
}

STYLE_MODELS = {
    "sdxl-img2img": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLImg2ImgPipeline",
        "torch_dtype": "float16",
    },
    "sd-img2img": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionImg2ImgPipeline",
        "torch_dtype": "float16",
    },
}

# Style prompts mapping
STYLE_PROMPTS = {
    "oil_painting": "oil painting style, thick brushstrokes, vibrant colors, artistic, masterpiece",
    "watercolor": "watercolor painting style, soft edges, flowing colors, artistic, delicate",
    "anime": "anime style, cel shading, vibrant colors, japanese animation, detailed",
    "sketch": "pencil sketch, black and white, detailed linework, hand drawn, artistic",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dark atmosphere, sci-fi",
    "ghibli": "studio ghibli style, soft colors, whimsical, animated, magical",
    "pixel_art": "pixel art style, 8-bit, retro video game aesthetic, nostalgic",
    "pop_art": "pop art style, bold colors, comic book, andy warhol inspired",
    "impressionist": "impressionist painting, monet style, soft brushstrokes, light and color",
}


class DiffusionService:
    """Service for running diffusion models locally on GPU."""

    def __init__(self):
        """Initialize the service - models are loaded lazily."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Cache for loaded pipelines
        self._pipelines: Dict[str, Any] = {}

        print(f"DiffusionService initialized")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string with data URL prefix.

        Args:
            image: PIL Image to convert
            format: Output format (PNG, JPEG, WEBP). Defaults to PNG.

        Returns:
            Data URL string (e.g., "data:image/png;base64,...")
        """
        buffer = io.BytesIO()
        format_upper = format.upper()

        # JPEG doesn't support alpha channel, ensure RGB mode
        if format_upper == "JPEG" and image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(buffer, format=format_upper, quality=95)
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Map format to MIME type
        mime_types = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
        }
        mime_type = mime_types.get(format_upper, "image/png")

        return f"data:{mime_type};base64,{b64_string}"

    @staticmethod
    def base64_to_image(b64_string: str) -> Tuple[Image.Image, str]:
        """Convert base64 string to PIL Image.

        Returns:
            Tuple of (PIL Image, original format string)
        """
        original_format = "PNG"  # Default fallback

        # Handle data URL format (e.g., "data:image/png;base64,...")
        if "," in b64_string:
            # Extract format from data URL
            header = b64_string.split(",")[0]
            if "image/jpeg" in header or "image/jpg" in header:
                original_format = "JPEG"
            elif "image/png" in header:
                original_format = "PNG"
            elif "image/webp" in header:
                original_format = "WEBP"
            b64_string = b64_string.split(",")[1]

        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data))

        # Also check PIL's detected format as fallback
        if image.format:
            original_format = image.format

        return image.convert("RGB"), original_format

    @staticmethod
    def base64_to_mask(b64_string: str) -> Image.Image:
        """Convert base64 string to grayscale mask."""
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data)).convert("L")

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float32":
            return torch.float32
        return torch.float16

    # =========================================================================
    # Pipeline Loading
    # =========================================================================

    def _load_pipeline(self, pipeline_key: str, config: dict) -> Any:
        """Load a pipeline if not already cached."""
        if pipeline_key in self._pipelines:
            return self._pipelines[pipeline_key]

        print(f"Loading pipeline: {pipeline_key} ({config['model_id']})")
        start_time = time.time()

        # Import the specific pipeline class
        from diffusers import (
            StableDiffusionInpaintPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
        )

        pipeline_classes = {
            "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
            "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
            "StableDiffusionInstructPix2PixPipeline": StableDiffusionInstructPix2PixPipeline,
            "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
            "StableDiffusionXLImg2ImgPipeline": StableDiffusionXLImg2ImgPipeline,
        }

        # Handle special cases
        if config["pipeline"] == "KandinskyV22InpaintPipeline":
            from diffusers import KandinskyV22InpaintPipeline
            pipeline_classes["KandinskyV22InpaintPipeline"] = KandinskyV22InpaintPipeline
        elif config["pipeline"] == "FluxFillPipeline":
            from diffusers import FluxFillPipeline
            pipeline_classes["FluxFillPipeline"] = FluxFillPipeline

        pipeline_class = pipeline_classes.get(config["pipeline"])
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline class: {config['pipeline']}")

        torch_dtype = self._get_torch_dtype(config["torch_dtype"])

        pipe = pipeline_class.from_pretrained(
            config["model_id"],
            torch_dtype=torch_dtype,
        )
        pipe = pipe.to(self.device)

        # Enable memory optimizations
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        self._pipelines[pipeline_key] = pipe
        print(f"  Loaded in {time.time() - start_time:.1f}s")

        return pipe

    def unload_pipeline(self, pipeline_key: str):
        """Unload a pipeline to free GPU memory."""
        if pipeline_key in self._pipelines:
            del self._pipelines[pipeline_key]
            torch.cuda.empty_cache()
            print(f"Unloaded pipeline: {pipeline_key}")

    def unload_all(self):
        """Unload all pipelines to free GPU memory."""
        self._pipelines.clear()
        torch.cuda.empty_cache()
        print("Unloaded all pipelines")

    # =========================================================================
    # Inpainting
    # =========================================================================

    def inpaint(
        self,
        image_b64: str,
        mask_b64: str,
        prompt: str,
        model_key: str = "sd-inpainting",
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Perform inpainting on an image.

        Returns:
            Tuple of (result_b64, error_message, processing_time)
        """
        start_time = time.time()

        try:
            # Validate model
            if model_key not in INPAINTING_MODELS:
                raise ValueError(f"Unknown inpainting model: {model_key}")

            config = INPAINTING_MODELS[model_key]

            # Decode inputs (also captures original format)
            image, original_format = self.base64_to_image(image_b64)
            mask = self.base64_to_mask(mask_b64)

            # Store original dimensions for rescaling output
            original_size = image.size  # (width, height)

            # Resize to model's optimal dimensions
            if "sdxl" in model_key or "flux" in model_key:
                target_size = (1024, 1024)
            else:
                target_size = (512, 512)

            image = image.resize(target_size, Image.Resampling.LANCZOS)
            mask = mask.resize(target_size, Image.Resampling.LANCZOS)

            # Load pipeline
            pipe = self._load_pipeline(model_key, config)

            # Run inference
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).images[0]

            # Rescale result back to original dimensions
            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)

            # Encode result in original format
            result_b64 = self.image_to_base64(result, original_format)
            processing_time = time.time() - start_time

            return result_b64, None, processing_time

        except Exception as e:
            return None, str(e), time.time() - start_time

    # =========================================================================
    # Style Transfer
    # =========================================================================

    def style_transfer(
        self,
        image_b64: str,
        style: str,
        model_key: str = "sdxl-img2img",
        strength: float = 0.6,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Apply style transfer to an image.

        Returns:
            Tuple of (result_b64, error_message, processing_time)
        """
        start_time = time.time()

        try:
            # Decode input (also captures original format)
            image, original_format = self.base64_to_image(image_b64)

            # Store original dimensions for rescaling output
            original_size = image.size  # (width, height)

            # Resize based on model
            if "sdxl" in model_key:
                target_size = (1024, 1024)
            else:
                target_size = (512, 512)
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Get style prompt
            style_prompt = STYLE_PROMPTS.get(style, style)

            # Select model
            if model_key not in STYLE_MODELS:
                model_key = "sdxl-img2img"
            config = STYLE_MODELS[model_key]

            # Load pipeline
            pipe = self._load_pipeline(f"style-{model_key}", config)

            # Run inference
            with torch.inference_mode():
                result = pipe(
                    prompt=style_prompt,
                    image=image,
                    strength=strength,
                ).images[0]

            # Rescale result back to original dimensions
            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)

            # Encode result in original format
            result_b64 = self.image_to_base64(result, original_format)
            processing_time = time.time() - start_time

            return result_b64, None, processing_time

        except Exception as e:
            return None, str(e), time.time() - start_time

    def get_style_presets(self) -> dict:
        """Get all available style presets."""
        return STYLE_PROMPTS.copy()

    def get_gpu_info(self) -> dict:
        """Get GPU information for health checks."""
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
                "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
            }
        return {"available": False}


# =============================================================================
# Singleton Instance
# =============================================================================

_diffusion_service: Optional[DiffusionService] = None


def get_diffusion_service() -> DiffusionService:
    """Get or create the DiffusionService singleton."""
    global _diffusion_service
    if _diffusion_service is None:
        _diffusion_service = DiffusionService()
    return _diffusion_service
