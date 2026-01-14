"""
Test script for style transfer using diffusion models.
Covers: artistic style transfer, ControlNet-based style, IP-Adapter style.

Usage:
    python test_style_transfer.py --image path/to/image.jpg --style "oil painting"
"""

import os
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
import base64
from PIL import Image
import io

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Style transfer related models
STYLE_MODELS = {
    # ControlNet models for structure preservation
    "controlnet-canny": "lllyasviel/sd-controlnet-canny",
    "controlnet-lineart": "lllyasviel/control_v11p_sd15_lineart",
    "controlnet-softedge": "lllyasviel/control_v11p_sd15_softedge",

    # SDXL for higher quality
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

# Predefined style prompts
STYLE_PRESETS = {
    "oil_painting": "oil painting style, thick brushstrokes, vibrant colors, artistic",
    "watercolor": "watercolor painting style, soft edges, flowing colors, artistic",
    "anime": "anime style, cel shading, vibrant colors, japanese animation",
    "sketch": "pencil sketch, black and white, detailed linework, hand drawn",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dark atmosphere",
    "ghibli": "studio ghibli style, soft colors, whimsical, animated",
    "pixel_art": "pixel art style, 8-bit, retro video game aesthetic",
    "pop_art": "pop art style, bold colors, comic book, andy warhol inspired",
    "impressionist": "impressionist painting, monet style, soft brushstrokes, light and color",
    "3d_render": "3d rendered, octane render, highly detailed, realistic lighting",
}


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_edges(image_path: str, output_path: str = None) -> Image.Image:
    """
    Extract canny edges from image for ControlNet.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("Install opencv: pip install opencv-python")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    result = Image.fromarray(edges_rgb)

    if output_path:
        result.save(output_path)
        print(f"Edges saved to {output_path}")

    return result


def style_transfer_with_controlnet_local(
    image_path: str,
    style_prompt: str,
    control_type: str = "canny",
    negative_prompt: str = "blurry, low quality, distorted, deformed",
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.8,
) -> Image.Image:
    """
    Style transfer using ControlNet for structure preservation.
    The ControlNet maintains the structure while the style prompt changes the appearance.
    """
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("Install: pip install diffusers transformers accelerate opencv-python")

    # Load ControlNet model
    control_model_id = STYLE_MODELS.get(f"controlnet-{control_type}")
    if not control_model_id:
        raise ValueError(f"Unknown control type: {control_type}")

    print(f"Loading ControlNet ({control_type})...")
    controlnet = ControlNetModel.from_pretrained(
        control_model_id,
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # Save VRAM

    # Prepare control image (edges)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # Extract edges for canny controlnet
    img_np = np.array(image)
    edges = cv2.Canny(img_np, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    control_image = Image.fromarray(edges)

    print(f"Applying style: '{style_prompt}'...")
    result = pipe(
        prompt=style_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    return result


def style_transfer_with_img2img_local(
    image_path: str,
    style_prompt: str,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    negative_prompt: str = "blurry, low quality, distorted",
) -> Image.Image:
    """
    Simple style transfer using img2img.
    Lower strength = more original image preserved.
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate")

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    print(f"Applying style: '{style_prompt}'...")
    result = pipe(
        prompt=style_prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images[0]

    return result


def style_transfer_with_ip_adapter_local(
    content_image_path: str,
    style_image_path: str,
    prompt: str = "",
    scale: float = 0.6,
) -> Image.Image:
    """
    Style transfer using IP-Adapter: transfer style from a reference image.
    This allows using an actual image as a style reference.
    """
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate")

    print("Loading Stable Diffusion with IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Load IP-Adapter
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(scale)
    pipe = pipe.to("cuda")

    content_image = Image.open(content_image_path).convert("RGB")
    style_image = Image.open(style_image_path).convert("RGB")

    print("Transferring style...")
    result = pipe(
        prompt=prompt,
        ip_adapter_image=style_image,
        image=content_image,
    ).images[0]

    return result


def batch_style_test(image_path: str, output_dir: str = "style_outputs"):
    """
    Test multiple style presets on a single image.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running batch style test with {len(STYLE_PRESETS)} styles...")

    for i, (style_name, style_prompt) in enumerate(STYLE_PRESETS.items()):
        print(f"\n[{i+1}/{len(STYLE_PRESETS)}] {style_name}")
        try:
            result = style_transfer_with_img2img_local(image_path, style_prompt, strength=0.65)
            output_path = os.path.join(output_dir, f"style_{style_name}.png")
            result.save(output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test style transfer models")
    parser.add_argument("--image", type=str, help="Path to content image")
    parser.add_argument("--style", type=str, default="oil_painting",
                       help=f"Style preset or custom prompt. Presets: {list(STYLE_PRESETS.keys())}")
    parser.add_argument("--style-image", type=str, help="Reference style image (for IP-Adapter mode)")
    parser.add_argument("--output", type=str, default="output_style.png", help="Output path")
    parser.add_argument("--mode", type=str, default="img2img",
                       choices=["img2img", "controlnet", "ip-adapter"], help="Style transfer method")
    parser.add_argument("--strength", type=float, default=0.6,
                       help="Style strength (0.0-1.0, higher = more stylized)")
    parser.add_argument("--batch-test", action="store_true", help="Test all style presets")
    parser.add_argument("--list-styles", action="store_true", help="List available style presets")

    args = parser.parse_args()

    if args.list_styles:
        print("Available style presets:")
        for name, prompt in STYLE_PRESETS.items():
            print(f"  {name}: {prompt[:50]}...")
        return

    if not args.image:
        print("No image provided. Running in demo mode...")
        print(f"\nStyle presets: {list(STYLE_PRESETS.keys())}")
        print("\nUsage examples:")
        print("  # Apply preset style:")
        print("  python test_style_transfer.py --image photo.jpg --style anime")
        print("\n  # Custom style prompt:")
        print("  python test_style_transfer.py --image photo.jpg --style 'van gogh starry night style'")
        print("\n  # Use ControlNet for better structure:")
        print("  python test_style_transfer.py --image photo.jpg --style anime --mode controlnet")
        print("\n  # Transfer style from another image:")
        print("  python test_style_transfer.py --image photo.jpg --style-image style_ref.jpg --mode ip-adapter")
        print("\n  # Batch test all styles:")
        print("  python test_style_transfer.py --image photo.jpg --batch-test")
        return

    if args.batch_test:
        batch_style_test(args.image)
        return

    # Get style prompt
    style_prompt = STYLE_PRESETS.get(args.style, args.style)

    # Run style transfer
    if args.mode == "controlnet":
        result = style_transfer_with_controlnet_local(args.image, style_prompt)
    elif args.mode == "ip-adapter":
        if not args.style_image:
            print("Error: --style-image required for ip-adapter mode")
            return
        result = style_transfer_with_ip_adapter_local(args.image, args.style_image)
    else:  # img2img
        result = style_transfer_with_img2img_local(args.image, style_prompt, strength=args.strength)

    if result:
        result.save(args.output)
        print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
