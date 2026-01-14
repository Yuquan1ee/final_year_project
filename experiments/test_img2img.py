"""

Test script for image-to-image editing models using HuggingFace.
Covers: image editing, background replacement, object modification.

Usage:
    python test_img2img.py --image path/to/image.jpg --prompt "make it sunset"
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Different models for different editing tasks
IMG2IMG_MODELS = {
    # Instruction-based editing
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",

    # Standard img2img
    "sd-img2img": "runwayml/stable-diffusion-v1-5",
    "sdxl-img2img": "stabilityai/stable-diffusion-xl-base-1.0",

    # Specialized editing
    "controlnet-canny": "lllyasviel/sd-controlnet-canny",
    "controlnet-depth": "lllyasviel/control_v11f1p_sd15_depth",

    # FLUX Kontext for image editing (recommended for API)
    "flux-kontext": "black-forest-labs/FLUX.1-Kontext-dev",
}


def edit_with_instruct_pix2pix_api(
    image_path: str,
    instruction: str,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Edit image using InstructPix2Pix via HuggingFace Inference API.
    This model understands natural language editing instructions.

    Args:
        image_path: Path to input image
        instruction: Natural language edit instruction (e.g., "make it winter", "add sunglasses")
        image_guidance_scale: How much to follow the original image (higher = more similar)
        guidance_scale: How much to follow the instruction
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN not set. Add it to .env file.")

    client = InferenceClient(token=HF_API_TOKEN)

    # Load image
    image = Image.open(image_path).convert("RGB")

    print(f"Editing with instruction: '{instruction}'...")
    try:
        result = client.image_to_image(
            image=image,
            prompt=instruction,
            model=IMG2IMG_MODELS["instruct-pix2pix"],
            guidance_scale=guidance_scale,
        )
        return result
    except Exception as e:
        print(f"Error with InstructPix2Pix: {e}")
        print("Trying FLUX Kontext model instead...")
        try:
            result = client.image_to_image(
                image=image,
                prompt=instruction,
                model=IMG2IMG_MODELS["flux-kontext"],
            )
            return result
        except Exception as e2:
            print(f"Error with FLUX Kontext: {e2}")
            return None


def edit_with_instruct_pix2pix_local(
    image_path: str,
    instruction: str,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> Image.Image:
    """
    Edit image using InstructPix2Pix locally (requires GPU).
    """
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        import torch
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate")

    print("Loading InstructPix2Pix model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IMG2IMG_MODELS["instruct-pix2pix"],
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    # Load and prepare image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    print(f"Editing with instruction: '{instruction}'...")
    result = pipe(
        instruction,
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
    ).images[0]

    return result


def img2img_with_sd_local(
    image_path: str,
    prompt: str,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    negative_prompt: str = "blurry, low quality",
) -> Image.Image:
    """
    Standard img2img with Stable Diffusion (local GPU).

    Args:
        strength: How much to transform the image (0.0 = no change, 1.0 = complete regeneration)
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate")

    print("Loading Stable Diffusion img2img model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        IMG2IMG_MODELS["sd-img2img"],
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    print(f"Transforming with prompt: '{prompt}'...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images[0]

    return result


def batch_edit_test(image_path: str, output_dir: str = "outputs"):
    """
    Test multiple editing instructions on a single image.
    Useful for comparing different edits.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_instructions = [
        "make it winter with snow",
        "make it sunset",
        "turn it into a painting",
        "add rain and clouds",
        "make it look vintage",
    ]

    print(f"Running batch test with {len(test_instructions)} instructions...")

    for i, instruction in enumerate(test_instructions):
        print(f"\n[{i+1}/{len(test_instructions)}] {instruction}")
        try:
            result = edit_with_instruct_pix2pix_api(image_path, instruction)
            if result:
                output_path = os.path.join(output_dir, f"edit_{i+1}_{instruction[:20].replace(' ', '_')}.png")
                result.save(output_path)
                print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test image editing models")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="make it sunset", help="Edit instruction/prompt")
    parser.add_argument("--output", type=str, default="output_edit.png", help="Output path")
    parser.add_argument("--local", action="store_true", help="Use local GPU instead of API")
    parser.add_argument("--mode", type=str, default="instruct",
                       choices=["instruct", "img2img"], help="Editing mode")
    parser.add_argument("--strength", type=float, default=0.75, help="Edit strength for img2img mode")
    parser.add_argument("--batch-test", action="store_true", help="Run batch test with multiple instructions")

    args = parser.parse_args()

    if not args.image:
        print("No image provided. Running in demo mode...")
        print(f"\nAvailable models: {list(IMG2IMG_MODELS.keys())}")
        print("\nUsage examples:")
        print("  # Instruction-based editing (recommended):")
        print("  python test_img2img.py --image photo.jpg --prompt 'make it winter'")
        print("\n  # Standard img2img with local GPU:")
        print("  python test_img2img.py --image photo.jpg --prompt 'sunset scene' --mode img2img --local")
        print("\n  # Batch test multiple edits:")
        print("  python test_img2img.py --image photo.jpg --batch-test")
        print("\nExample instructions for InstructPix2Pix:")
        print("  - 'make it winter'")
        print("  - 'add sunglasses'")
        print("  - 'turn into a watercolor painting'")
        print("  - 'make the sky dramatic'")
        print("  - 'replace background with beach'")
        return

    if args.batch_test:
        batch_edit_test(args.image)
        return

    # Run editing
    if args.mode == "instruct":
        if args.local:
            result = edit_with_instruct_pix2pix_local(args.image, args.prompt)
        else:
            result = edit_with_instruct_pix2pix_api(args.image, args.prompt)
    else:  # img2img
        result = img2img_with_sd_local(args.image, args.prompt, strength=args.strength)

    if result:
        result.save(args.output)
        print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
