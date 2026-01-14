"""
HuggingFace API integration service.
Handles all communication with HuggingFace Inference API.
"""

import os
import base64
import httpx
from PIL import Image
import io
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_BASE = "https://api-inference.huggingface.co/models"

# Model configurations
MODELS = {
    "inpainting": {
        "sd-inpainting": "runwayml/stable-diffusion-inpainting",
        "sdxl-inpainting": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "kandinsky-inpainting": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    },
    "editing": {
        "instruct-pix2pix": "timbrooks/instruct-pix2pix",
        "sd-img2img": "runwayml/stable-diffusion-v1-5",
    },
    "style": {
        "sd": "runwayml/stable-diffusion-v1-5",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }
}

# Style presets
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


class HuggingFaceService:
    """Service for interacting with HuggingFace Inference API."""

    def __init__(self):
        if not HF_API_TOKEN:
            raise ValueError("HF_API_TOKEN environment variable not set")
        self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        self.client = httpx.AsyncClient(timeout=120.0)

    async def _call_api(self, model_id: str, payload: dict) -> bytes:
        """Make API call to HuggingFace."""
        url = f"{HF_API_BASE}/{model_id}"
        response = await self.client.post(url, headers=self.headers, json=payload)

        if response.status_code != 200:
            error_msg = response.text
            raise Exception(f"HuggingFace API error ({response.status_code}): {error_msg}")

        return response.content

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def base64_to_image(b64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))

    async def inpaint(
        self,
        image_b64: str,
        mask_b64: str,
        prompt: str,
        model_key: str = "sd-inpainting",
        negative_prompt: str = "blurry, low quality",
        **kwargs
    ) -> str:
        """
        Perform inpainting using HuggingFace API.

        Returns: Base64 encoded result image
        """
        model_id = MODELS["inpainting"].get(model_key)
        if not model_id:
            raise ValueError(f"Unknown inpainting model: {model_key}")

        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "image": image_b64,
                "mask_image": mask_b64,
                **kwargs
            }
        }

        result_bytes = await self._call_api(model_id, payload)
        result_image = Image.open(io.BytesIO(result_bytes))
        return self.image_to_base64(result_image)

    async def edit_image(
        self,
        image_b64: str,
        instruction: str,
        mode: str = "instruct",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        strength: float = 0.75,
        **kwargs
    ) -> str:
        """
        Edit image using instruction or img2img.

        Returns: Base64 encoded result image
        """
        if mode == "instruct":
            model_id = MODELS["editing"]["instruct-pix2pix"]
            payload = {
                "inputs": instruction,
                "parameters": {
                    "image": image_b64,
                    "guidance_scale": guidance_scale,
                    "image_guidance_scale": image_guidance_scale,
                    **kwargs
                }
            }
        else:  # img2img
            model_id = MODELS["editing"]["sd-img2img"]
            payload = {
                "inputs": instruction,
                "parameters": {
                    "image": image_b64,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    **kwargs
                }
            }

        result_bytes = await self._call_api(model_id, payload)
        result_image = Image.open(io.BytesIO(result_bytes))
        return self.image_to_base64(result_image)

    async def style_transfer(
        self,
        image_b64: str,
        style: str,
        strength: float = 0.6,
        **kwargs
    ) -> str:
        """
        Apply style transfer to image.

        Args:
            style: Style preset key or custom prompt

        Returns: Base64 encoded result image
        """
        # Get style prompt
        style_prompt = STYLE_PROMPTS.get(style, style)

        model_id = MODELS["style"]["sd"]
        payload = {
            "inputs": style_prompt,
            "parameters": {
                "image": image_b64,
                "strength": strength,
                **kwargs
            }
        }

        result_bytes = await self._call_api(model_id, payload)
        result_image = Image.open(io.BytesIO(result_bytes))
        return self.image_to_base64(result_image)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
_service: Optional[HuggingFaceService] = None


def get_hf_service() -> HuggingFaceService:
    """Get or create HuggingFace service instance."""
    global _service
    if _service is None:
        _service = HuggingFaceService()
    return _service
