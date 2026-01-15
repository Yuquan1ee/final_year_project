#!/usr/bin/env python3
"""
Utility script to create masks for inpainting experiments.

Usage:
    # Create center mask (default)
    python create_mask.py --image photo.jpg

    # Create mask for specific region
    python create_mask.py --image photo.jpg --region left
    python create_mask.py --image photo.jpg --region right
    python create_mask.py --image photo.jpg --region top
    python create_mask.py --image photo.jpg --region bottom

    # Create mask with custom box (x1,y1,x2,y2 as percentages)
    python create_mask.py --image photo.jpg --box 0.2,0.2,0.8,0.8

    # Process all images in a directory
    python create_mask.py --input-dir ./test_images/inpainting
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def create_mask(image_path: str, region: str = "center", box: tuple = None) -> Image.Image:
    """
    Create a mask for inpainting.

    Args:
        image_path: Path to the input image
        region: Preset region ("center", "left", "right", "top", "bottom")
        box: Custom box as (x1, y1, x2, y2) percentages (0.0-1.0)

    Returns:
        Mask image (white = inpaint, black = keep)
    """
    img = Image.open(image_path)
    w, h = img.size

    # Create black image (keep everything by default)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if box:
        # Custom box as percentages
        x1, y1, x2, y2 = box
        pixel_box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    else:
        # Preset regions
        regions = {
            "center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
            "left": (0, h // 4, w // 3, 3 * h // 4),
            "right": (2 * w // 3, h // 4, w, 3 * h // 4),
            "top": (w // 4, 0, 3 * w // 4, h // 3),
            "bottom": (w // 4, 2 * h // 3, 3 * w // 4, h),
            "small_center": (3 * w // 8, 3 * h // 8, 5 * w // 8, 5 * h // 8),
        }
        pixel_box = regions.get(region, regions["center"])

    draw.rectangle(pixel_box, fill=255)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Create masks for inpainting")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--input-dir", type=str, help="Process all images in directory")
    parser.add_argument(
        "--region",
        type=str,
        default="center",
        choices=["center", "left", "right", "top", "bottom", "small_center"],
        help="Preset mask region",
    )
    parser.add_argument(
        "--box",
        type=str,
        help="Custom box as 'x1,y1,x2,y2' percentages (e.g., '0.2,0.2,0.8,0.8')",
    )
    parser.add_argument("--output", type=str, help="Output path (default: <image>_mask.png)")

    args = parser.parse_args()

    # Parse custom box if provided
    box = None
    if args.box:
        box = tuple(float(x) for x in args.box.split(","))
        if len(box) != 4:
            print("Error: --box must have 4 values: x1,y1,x2,y2")
            return

    if args.input_dir:
        # Process all images in directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return

        extensions = (".jpg", ".jpeg", ".png", ".webp")
        images = [
            f
            for f in input_dir.iterdir()
            if f.suffix.lower() in extensions and "_mask" not in f.stem
        ]

        print(f"Found {len(images)} images in {input_dir}")

        for img_path in images:
            mask = create_mask(str(img_path), args.region, box)
            output_path = img_path.parent / f"{img_path.stem}_mask.png"
            mask.save(output_path)
            print(f"Created: {output_path}")

    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            return

        mask = create_mask(args.image, args.region, box)

        if args.output:
            output_path = args.output
        else:
            output_path = img_path.parent / f"{img_path.stem}_mask.png"

        mask.save(output_path)
        print(f"Mask saved to: {output_path}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python create_mask.py --image photo.jpg")
        print("  python create_mask.py --image photo.jpg --region left")
        print("  python create_mask.py --input-dir ./test_images/inpainting")


if __name__ == "__main__":
    main()
