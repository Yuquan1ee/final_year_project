#!/usr/bin/env python3
"""
Consolidated experiment script for testing diffusion models on NTU HPC cluster.
Combines: Image-to-Image editing, Inpainting, and Style Transfer.

This script is designed to run on GPU nodes with local model inference.

Usage:
    # Run all experiments
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs

    # Run specific experiment type
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs --experiment img2img
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs --experiment inpainting
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs --experiment style

    # Run with specific image
    python run_all_experiments.py --image test.jpg --output-dir ./outputs

Required directory structure:
    test_images/
    ├── content/          # Images for img2img and style transfer
    │   └── *.jpg/png
    ├── inpainting/       # Images for inpainting
    │   ├── image.jpg     # Original images
    │   └── image_mask.png # Corresponding masks (white=inpaint, black=keep)
    └── style_refs/       # (Optional) Reference images for IP-Adapter
        └── *.jpg/png
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Configurations
# ============================================================================

IMG2IMG_MODELS = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "sd-v1.5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

INPAINTING_MODELS = {
    "sd-inpainting": "runwayml/stable-diffusion-inpainting",
    "sdxl-inpainting": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "kandinsky-inpainting": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}

CONTROLNET_MODELS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
}

STYLE_PRESETS = {
    "oil_painting": "oil painting style, thick brushstrokes, vibrant colors, artistic",
    "watercolor": "watercolor painting style, soft edges, flowing colors, artistic",
    "anime": "anime style, cel shading, vibrant colors, japanese animation",
    "sketch": "pencil sketch, black and white, detailed linework, hand drawn",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dark atmosphere",
    "ghibli": "studio ghibli style, soft colors, whimsical, animated",
    "impressionist": "impressionist painting, monet style, soft brushstrokes, light and color",
    "3d_render": "3d rendered, octane render, highly detailed, realistic lighting",
}

EDIT_INSTRUCTIONS = [
    "make it snow",
]


# ============================================================================
# Utility Functions
# ============================================================================

def setup_torch():
    """Setup PyTorch with proper device configuration."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        logger.warning("CUDA not available, using CPU (this will be very slow)")

    return device


def find_images(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """Find all image files in a directory."""
    images = []
    if directory.exists():
        for ext in extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(images)


def create_output_dir(base_dir: Path, experiment_type: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{experiment_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_experiment_metadata(output_dir: Path, metadata: Dict[str, Any]):
    """Save experiment metadata to JSON."""
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")


def create_sample_mask(image: Image.Image, region: str = "center") -> Image.Image:
    """
    Create a sample mask for testing inpainting.

    Args:
        image: Input PIL Image
        region: "center", "left", "right", "top", "bottom"

    Returns:
        Mask image (white = inpaint, black = keep)
    """
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    regions = {
        "center": (w//4, h//4, 3*w//4, 3*h//4),
        "left": (0, h//4, w//3, 3*h//4),
        "right": (2*w//3, h//4, w, 3*h//4),
        "top": (w//4, 0, 3*w//4, h//3),
        "bottom": (w//4, 2*h//3, 3*w//4, h),
    }

    box = regions.get(region, regions["center"])
    draw.rectangle(box, fill=255)

    return mask


# ============================================================================
# Image-to-Image Experiments
# ============================================================================

def run_img2img_experiments(
    images: List[Path],
    output_dir: Path,
    models: List[str] = None,
    instructions: List[str] = None,
):
    """
    Run image-to-image editing experiments.

    Tests:
    1. InstructPix2Pix - instruction-based editing
    2. Stable Diffusion img2img - prompt-based transformation
    """
    import torch
    from diffusers import (
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionImg2ImgPipeline,
    )

    device = setup_torch()
    models = models or ["instruct-pix2pix", "sd-v1.5"]
    instructions = instructions or EDIT_INSTRUCTIONS

    results = []

    # Test InstructPix2Pix
    if "instruct-pix2pix" in models:
        logger.info("=" * 60)
        logger.info("Testing InstructPix2Pix")
        logger.info("=" * 60)

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            IMG2IMG_MODELS["instruct-pix2pix"],
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        for img_path in images:
            image = Image.open(img_path).convert("RGB").resize((512, 512))
            img_name = img_path.stem

            for instruction in instructions:
                logger.info(f"Processing {img_name}: '{instruction}'")
                start_time = time.time()

                try:
                    result = pipe(
                        instruction,
                        image=image,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        image_guidance_scale=1.5,
                    ).images[0]

                    elapsed = time.time() - start_time
                    safe_instruction = instruction[:30].replace(' ', '_').replace('/', '-')
                    output_path = output_dir / f"pix2pix_{img_name}_{safe_instruction}.png"
                    result.save(output_path)

                    results.append({
                        "model": "instruct-pix2pix",
                        "input": str(img_path),
                        "instruction": instruction,
                        "output": str(output_path),
                        "time_seconds": elapsed,
                        "status": "success",
                    })
                    logger.info(f"  Saved: {output_path} ({elapsed:.1f}s)")

                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        "model": "instruct-pix2pix",
                        "input": str(img_path),
                        "instruction": instruction,
                        "status": "error",
                        "error": str(e),
                    })

        del pipe
        torch.cuda.empty_cache()

    # Test SD img2img
    if "sd-v1.5" in models:
        logger.info("=" * 60)
        logger.info("Testing Stable Diffusion img2img")
        logger.info("=" * 60)

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            IMG2IMG_MODELS["sd-v1.5"],
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        test_prompts = [
            ("sunset", "beautiful sunset scene, golden hour lighting"),
            ("winter", "winter scene with snow, cold atmosphere"),
            ("night", "nighttime scene, moonlight, stars"),
        ]

        for img_path in images:
            image = Image.open(img_path).convert("RGB").resize((512, 512))
            img_name = img_path.stem

            for prompt_name, prompt in test_prompts:
                for strength in [0.5, 0.75]:
                    logger.info(f"Processing {img_name}: {prompt_name} (strength={strength})")
                    start_time = time.time()

                    try:
                        result = pipe(
                            prompt=prompt,
                            negative_prompt="blurry, low quality, distorted",
                            image=image,
                            strength=strength,
                            guidance_scale=7.5,
                        ).images[0]

                        elapsed = time.time() - start_time
                        output_path = output_dir / f"sd_img2img_{img_name}_{prompt_name}_s{int(strength*100)}.png"
                        result.save(output_path)

                        results.append({
                            "model": "sd-v1.5-img2img",
                            "input": str(img_path),
                            "prompt": prompt,
                            "strength": strength,
                            "output": str(output_path),
                            "time_seconds": elapsed,
                            "status": "success",
                        })
                        logger.info(f"  Saved: {output_path} ({elapsed:.1f}s)")

                    except Exception as e:
                        logger.error(f"  Error: {e}")
                        results.append({
                            "model": "sd-v1.5-img2img",
                            "input": str(img_path),
                            "prompt": prompt,
                            "strength": strength,
                            "status": "error",
                            "error": str(e),
                        })

        del pipe
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Inpainting Experiments
# ============================================================================

def run_inpainting_experiments(
    images: List[Path],
    masks: Dict[str, Path],
    output_dir: Path,
    models: List[str] = None,
):
    """
    Run inpainting experiments.

    Tests multiple inpainting models with various prompts.
    """
    import torch
    from diffusers import AutoPipelineForInpainting

    device = setup_torch()
    models = models or ["sd-inpainting"]

    inpaint_prompts = [
        "remove the shadow",
    ]

    results = []

    for model_key in models:
        if model_key not in INPAINTING_MODELS:
            logger.warning(f"Unknown model: {model_key}")
            continue

        logger.info("=" * 60)
        logger.info(f"Testing {model_key}")
        logger.info("=" * 60)

        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                INPAINTING_MODELS[model_key],
                torch_dtype=torch.float16,
            ).to(device)
        except Exception as e:
            logger.error(f"Failed to load {model_key}: {e}")
            continue

        for img_path in images:
            img_name = img_path.stem
            image = Image.open(img_path).convert("RGB")

            # Find or create mask
            mask_path = masks.get(img_name)
            if mask_path and mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                logger.info(f"Using provided mask: {mask_path}")
            else:
                mask = create_sample_mask(image, "center")
                logger.info(f"Using auto-generated center mask")

            # Resize to standard size
            image = image.resize((512, 512))
            mask = mask.resize((512, 512))

            for prompt in inpaint_prompts:
                logger.info(f"Processing {img_name}: '{prompt}'")
                start_time = time.time()

                try:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt="blurry, low quality, distorted, deformed",
                        image=image,
                        mask_image=mask,
                        num_inference_steps=30,
                    ).images[0]

                    elapsed = time.time() - start_time
                    safe_prompt = prompt[:25].replace(' ', '_').replace('/', '-')
                    output_path = output_dir / f"{model_key}_{img_name}_{safe_prompt}.png"
                    result.save(output_path)

                    results.append({
                        "model": model_key,
                        "input": str(img_path),
                        "mask": str(mask_path) if mask_path else "auto-generated",
                        "prompt": prompt,
                        "output": str(output_path),
                        "time_seconds": elapsed,
                        "status": "success",
                    })
                    logger.info(f"  Saved: {output_path} ({elapsed:.1f}s)")

                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        "model": model_key,
                        "input": str(img_path),
                        "prompt": prompt,
                        "status": "error",
                        "error": str(e),
                    })

        del pipe
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Style Transfer Experiments
# ============================================================================

def run_style_transfer_experiments(
    images: List[Path],
    output_dir: Path,
    styles: List[str] = None,
    use_controlnet: bool = True,
):
    """
    Run style transfer experiments.

    Tests:
    1. img2img-based style transfer
    2. ControlNet-based style transfer (preserves structure)
    """
    import torch
    import numpy as np
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionControlNetPipeline,
        ControlNetModel,
    )

    device = setup_torch()
    styles = styles or list(STYLE_PRESETS.keys())

    results = []

    # Test img2img style transfer
    logger.info("=" * 60)
    logger.info("Testing img2img Style Transfer")
    logger.info("=" * 60)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    for img_path in images:
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        img_name = img_path.stem

        for style_name in styles:
            style_prompt = STYLE_PRESETS.get(style_name, style_name)

            for strength in [0.5, 0.7]:
                logger.info(f"Processing {img_name}: {style_name} (strength={strength})")
                start_time = time.time()

                try:
                    result = pipe(
                        prompt=style_prompt,
                        negative_prompt="blurry, low quality, distorted",
                        image=image,
                        strength=strength,
                        guidance_scale=7.5,
                    ).images[0]

                    elapsed = time.time() - start_time
                    output_path = output_dir / f"style_img2img_{img_name}_{style_name}_s{int(strength*100)}.png"
                    result.save(output_path)

                    results.append({
                        "model": "sd-v1.5-img2img",
                        "method": "img2img",
                        "input": str(img_path),
                        "style": style_name,
                        "strength": strength,
                        "output": str(output_path),
                        "time_seconds": elapsed,
                        "status": "success",
                    })
                    logger.info(f"  Saved: {output_path} ({elapsed:.1f}s)")

                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        "model": "sd-v1.5-img2img",
                        "method": "img2img",
                        "input": str(img_path),
                        "style": style_name,
                        "strength": strength,
                        "status": "error",
                        "error": str(e),
                    })

    del pipe
    torch.cuda.empty_cache()

    # Test ControlNet style transfer
    if use_controlnet:
        logger.info("=" * 60)
        logger.info("Testing ControlNet Style Transfer")
        logger.info("=" * 60)

        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, skipping ControlNet tests")
            return results

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODELS["canny"],
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        pipe.enable_model_cpu_offload()

        for img_path in images:
            image = Image.open(img_path).convert("RGB").resize((512, 512))
            img_name = img_path.stem

            # Extract canny edges
            img_np = np.array(image)
            edges = cv2.Canny(img_np, 100, 200)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            control_image = Image.fromarray(edges)

            # Save edge image for reference
            edge_path = output_dir / f"edges_{img_name}.png"
            control_image.save(edge_path)

            for style_name in styles[:4]:  # Test subset for ControlNet (slower)
                style_prompt = STYLE_PRESETS.get(style_name, style_name)

                logger.info(f"Processing {img_name}: {style_name} (ControlNet)")
                start_time = time.time()

                try:
                    result = pipe(
                        prompt=style_prompt,
                        negative_prompt="blurry, low quality, distorted, deformed",
                        image=control_image,
                        num_inference_steps=30,
                        controlnet_conditioning_scale=0.8,
                    ).images[0]

                    elapsed = time.time() - start_time
                    output_path = output_dir / f"style_controlnet_{img_name}_{style_name}.png"
                    result.save(output_path)

                    results.append({
                        "model": "controlnet-canny",
                        "method": "controlnet",
                        "input": str(img_path),
                        "style": style_name,
                        "output": str(output_path),
                        "time_seconds": elapsed,
                        "status": "success",
                    })
                    logger.info(f"  Saved: {output_path} ({elapsed:.1f}s)")

                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        "model": "controlnet-canny",
                        "method": "controlnet",
                        "input": str(img_path),
                        "style": style_name,
                        "status": "error",
                        "error": str(e),
                    })

        del pipe, controlnet
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidated diffusion model experiments for HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments with test images directory
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs

    # Run only img2img experiments
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs --experiment img2img

    # Run with a single image
    python run_all_experiments.py --image test.jpg --output-dir ./outputs

    # Run specific styles only
    python run_all_experiments.py --input-dir ./test_images --output-dir ./outputs --experiment style --styles anime,oil_painting
        """
    )

    parser.add_argument("--input-dir", type=str, help="Directory containing test images")
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment", type=str, choices=["all", "img2img", "inpainting", "style"],
                       default="all", help="Which experiment to run")
    parser.add_argument("--styles", type=str, help="Comma-separated list of styles to test")
    parser.add_argument("--no-controlnet", action="store_true", help="Skip ControlNet experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without running")

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir and not args.image:
        parser.print_help()
        print("\n" + "=" * 60)
        print("ERROR: Please provide --input-dir or --image")
        print("=" * 60)
        print("\nExpected directory structure for --input-dir:")
        print("  test_images/")
        print("  ├── content/           # Images for img2img and style transfer")
        print("  ├── inpainting/        # Images for inpainting")
        print("  │   ├── image.jpg")
        print("  │   └── image_mask.png  # Optional masks")
        print("  └── style_refs/        # Optional style reference images")
        return

    # Setup paths
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Collect images
    content_images = []
    inpaint_images = []
    inpaint_masks = {}

    if args.image:
        img_path = Path(args.image)
        if img_path.exists():
            content_images = [img_path]
            inpaint_images = [img_path]
        else:
            logger.error(f"Image not found: {args.image}")
            return
    elif args.input_dir:
        input_dir = Path(args.input_dir)

        # Content images
        content_dir = input_dir / "content"
        if content_dir.exists():
            content_images = find_images(content_dir)
        else:
            # Fall back to root directory
            content_images = find_images(input_dir)

        # Inpainting images and masks
        inpaint_dir = input_dir / "inpainting"
        if inpaint_dir.exists():
            for img in find_images(inpaint_dir):
                if "_mask" not in img.stem:
                    inpaint_images.append(img)
                    # Look for corresponding mask
                    mask_path = inpaint_dir / f"{img.stem}_mask.png"
                    if mask_path.exists():
                        inpaint_masks[img.stem] = mask_path
        else:
            inpaint_images = content_images

    # Parse styles
    styles = None
    if args.styles:
        styles = [s.strip() for s in args.styles.split(",")]

    # Log configuration
    logger.info("=" * 60)
    logger.info("Experiment Configuration")
    logger.info("=" * 60)
    logger.info(f"Experiment type: {args.experiment}")
    logger.info(f"Content images: {len(content_images)}")
    logger.info(f"Inpainting images: {len(inpaint_images)}")
    logger.info(f"Inpainting masks: {len(inpaint_masks)}")
    logger.info(f"Output directory: {output_base}")
    if styles:
        logger.info(f"Styles to test: {styles}")

    if args.dry_run:
        logger.info("\nDry run - no experiments will be executed")
        return

    if not content_images and not inpaint_images:
        logger.error("No images found to process")
        return

    # Run experiments
    all_results = []
    start_time = time.time()

    if args.experiment in ["all", "img2img"] and content_images:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING IMAGE-TO-IMAGE EXPERIMENTS")
        logger.info("=" * 60)
        output_dir = create_output_dir(output_base, "img2img")
        results = run_img2img_experiments(content_images, output_dir)
        all_results.extend(results)
        save_experiment_metadata(output_dir, {
            "experiment_type": "img2img",
            "images": [str(p) for p in content_images],
            "results": results,
        })

    if args.experiment in ["all", "inpainting"] and inpaint_images:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING INPAINTING EXPERIMENTS")
        logger.info("=" * 60)
        output_dir = create_output_dir(output_base, "inpainting")
        results = run_inpainting_experiments(inpaint_images, inpaint_masks, output_dir)
        all_results.extend(results)
        save_experiment_metadata(output_dir, {
            "experiment_type": "inpainting",
            "images": [str(p) for p in inpaint_images],
            "masks": {k: str(v) for k, v in inpaint_masks.items()},
            "results": results,
        })

    if args.experiment in ["all", "style"] and content_images:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING STYLE TRANSFER EXPERIMENTS")
        logger.info("=" * 60)
        output_dir = create_output_dir(output_base, "style")
        results = run_style_transfer_experiments(
            content_images,
            output_dir,
            styles=styles,
            use_controlnet=not args.no_controlnet,
        )
        all_results.extend(results)
        save_experiment_metadata(output_dir, {
            "experiment_type": "style_transfer",
            "images": [str(p) for p in content_images],
            "styles": styles or list(STYLE_PRESETS.keys()),
            "results": results,
        })

    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get("status") == "success")
    failed = sum(1 for r in all_results if r.get("status") == "error")

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Output directory: {output_base}")

    # Save overall summary
    summary_path = output_base / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_experiments": len(all_results),
            "successful": successful,
            "failed": failed,
            "total_time_seconds": total_time,
            "results": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
