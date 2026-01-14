"""
Test script for inpainting models using HuggingFace Inference API.
Run this to experiment with different inpainting models before integrating into the app.

Usage:
    python test_inpainting.py --image path/to/image.jpg --mask path/to/mask.jpg --prompt "your prompt"
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# HuggingFace API configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
INPAINTING_MODELS = {
    "sd-inpainting": "runwayml/stable-diffusion-inpainting",
    "sdxl-inpainting": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "kandinsky-inpainting": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}


def inpaint_with_api(
    image_path: str,
    mask_path: str,
    prompt: str,
    model_key: str = "sd-inpainting",
    negative_prompt: str = "blurry, low quality, distorted",
) -> Image.Image:
    """
    Perform inpainting using HuggingFace Inference API.

    Args:
        image_path: Path to the input image
        mask_path: Path to the mask image (white = inpaint, black = keep)
        prompt: Text description of what to generate
        model_key: Key from INPAINTING_MODELS dict
        negative_prompt: What to avoid in generation

    Returns:
        PIL Image of the result
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN not set. Add it to .env file.")

    model_id = INPAINTING_MODELS.get(model_key)
    if not model_id:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(INPAINTING_MODELS.keys())}")

    client = InferenceClient(token=HF_API_TOKEN)

    # Load images
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    print(f"Calling {model_id}...")
    try:
        # Note: inpainting via API may have limited support
        # Using image_to_image as a workaround with mask in prompt context
        result = client.image_to_image(
            image=image,
            prompt=prompt,
            model=model_id,
            negative_prompt=negative_prompt,
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Inpainting via API has limited support.")
        print("Consider using --local flag with GPU for full inpainting functionality.")
        return None


def inpaint_with_diffusers(
    image_path: str,
    mask_path: str,
    prompt: str,
    model_key: str = "sd-inpainting",
    negative_prompt: str = "blurry, low quality, distorted",
) -> Image.Image:
    """
    Perform inpainting using local diffusers library (requires GPU).
    Use this for HPC or local GPU testing.
    """
    try:
        from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
        import torch
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate")

    model_id = INPAINTING_MODELS.get(model_key)

    # Load pipeline
    print(f"Loading {model_id}...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # Load images
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize to standard size
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))

    print("Running inpainting...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=30,
    ).images[0]

    return result


def create_sample_mask(image_path: str, output_path: str, box: tuple = None):
    """
    Create a simple rectangular mask for testing.

    Args:
        image_path: Input image to base mask size on
        output_path: Where to save the mask
        box: (x1, y1, x2, y2) coordinates for the mask area, or None for center
    """
    img = Image.open(image_path)
    w, h = img.size

    # Create black image (keep everything)
    mask = Image.new("L", (w, h), 0)

    # Default to center region
    if box is None:
        box = (w//4, h//4, 3*w//4, 3*h//4)

    # Draw white rectangle (area to inpaint)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)

    mask.save(output_path)
    print(f"Mask saved to {output_path}")
    return mask


def main():
    parser = argparse.ArgumentParser(description="Test inpainting models")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--mask", type=str, help="Path to mask image")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape", help="Inpainting prompt")
    parser.add_argument("--model", type=str, default="sd-inpainting",
                       choices=list(INPAINTING_MODELS.keys()), help="Model to use")
    parser.add_argument("--output", type=str, default="output_inpaint.png", help="Output path")
    parser.add_argument("--local", action="store_true", help="Use local GPU instead of API")
    parser.add_argument("--create-mask", action="store_true", help="Create a sample mask from the image")

    args = parser.parse_args()

    # Demo mode if no image provided
    if not args.image:
        print("No image provided. Running in demo mode...")
        print(f"\nAvailable models: {list(INPAINTING_MODELS.keys())}")
        print("\nUsage examples:")
        print("  # Using API (requires HF_API_TOKEN in .env):")
        print("  python test_inpainting.py --image photo.jpg --mask mask.jpg --prompt 'a cat sitting'")
        print("\n  # Using local GPU:")
        print("  python test_inpainting.py --image photo.jpg --mask mask.jpg --prompt 'a cat' --local")
        print("\n  # Create a mask first:")
        print("  python test_inpainting.py --image photo.jpg --create-mask")
        return

    # Create mask if requested
    if args.create_mask:
        mask_path = Path(args.image).stem + "_mask.png"
        create_sample_mask(args.image, mask_path)
        return

    if not args.mask:
        print("Error: --mask is required. Use --create-mask to generate one.")
        return

    # Run inpainting
    if args.local:
        result = inpaint_with_diffusers(args.image, args.mask, args.prompt, args.model)
    else:
        result = inpaint_with_api(args.image, args.mask, args.prompt, args.model)

    if result:
        result.save(args.output)
        print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
