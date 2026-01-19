#!/usr/bin/env python3
"""
Interactive mask creation tool for inpainting experiments.

Usage:
    python interactive_mask.py --input-dir ./test_images/inpainting
    python interactive_mask.py --image ./test_images/inpainting/photo.jpg

Controls:
    - Left click + drag: Draw mask region
    - 'r': Reset current mask (start over)
    - 'u': Undo last drawn region
    - 's' or Enter: Save mask and move to next image
    - 'q' or Escape: Quit without saving current image
    - 'b': Toggle brush size (small/medium/large)
    - 'm': Toggle mode (rectangle/freehand)
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import matplotlib

# Use TkAgg backend for interactive display
matplotlib.use('TkAgg')


class InteractiveMaskCreator:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = np.array(Image.open(image_path))
        self.height, self.width = self.image.shape[:2]

        # Mask (0 = keep, 255 = inpaint)
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask_history = []  # For undo functionality

        # Drawing state
        self.drawing = False
        self.mode = 'freehand'  # 'rectangle' or 'freehand'
        self.brush_sizes = [5, 15, 30]
        self.brush_idx = 1
        self.brush_size = self.brush_sizes[self.brush_idx]

        # Rectangle drawing
        self.rect_start = None
        self.current_rect = None

        # Freehand points
        self.freehand_points = []

        # Result
        self.saved = False

        # Setup figure
        self.setup_figure()

    def setup_figure(self):
        """Setup matplotlib figure and events."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f'Mask Creator - {self.image_path.name}')

        # Display image
        self.img_display = self.ax.imshow(self.image)

        # Overlay for mask (red tint)
        self.mask_overlay = self.ax.imshow(
            np.zeros((self.height, self.width, 4), dtype=np.uint8),
            alpha=0.5
        )

        self.ax.set_title(
            f"Mode: {self.mode.upper()} | Brush: {self.brush_size}px\n"
            "Draw regions to inpaint | 'm'=toggle mode | 'b'=brush size | 's'=save | 'r'=reset | 'u'=undo | 'q'=quit"
        )
        self.ax.axis('off')

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()

    def update_overlay(self):
        """Update the mask overlay display."""
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        overlay[self.mask > 0] = [255, 0, 0, 128]  # Red with alpha
        self.mask_overlay.set_data(overlay)
        self.fig.canvas.draw_idle()

    def update_title(self):
        """Update the title with current mode and brush size."""
        self.ax.set_title(
            f"Mode: {self.mode.upper()} | Brush: {self.brush_size}px\n"
            "Draw regions to inpaint | 'm'=toggle mode | 'b'=brush size | 's'=save | 'r'=reset | 'u'=undo | 'q'=quit"
        )
        self.fig.canvas.draw_idle()

    def save_mask_state(self):
        """Save current mask state for undo."""
        self.mask_history.append(self.mask.copy())
        # Keep only last 20 states
        if len(self.mask_history) > 20:
            self.mask_history.pop(0)

    def on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.ax or event.button != 1:
            return

        self.drawing = True
        x, y = int(event.xdata), int(event.ydata)

        if self.mode == 'rectangle':
            self.rect_start = (x, y)
            self.current_rect = Rectangle(
                (x, y), 0, 0,
                linewidth=2, edgecolor='white', facecolor='red', alpha=0.3
            )
            self.ax.add_patch(self.current_rect)
        else:  # freehand
            self.save_mask_state()
            self.freehand_points = [(x, y)]
            self.draw_brush(x, y)

    def on_motion(self, event):
        """Handle mouse motion."""
        if not self.drawing or event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.mode == 'rectangle' and self.rect_start:
            # Update rectangle
            x0, y0 = self.rect_start
            self.current_rect.set_xy((min(x0, x), min(y0, y)))
            self.current_rect.set_width(abs(x - x0))
            self.current_rect.set_height(abs(y - y0))
            self.fig.canvas.draw_idle()
        else:  # freehand
            self.freehand_points.append((x, y))
            self.draw_brush(x, y)

    def on_release(self, event):
        """Handle mouse release."""
        if not self.drawing:
            return

        self.drawing = False

        if self.mode == 'rectangle' and self.rect_start and event.inaxes == self.ax:
            self.save_mask_state()
            x, y = int(event.xdata), int(event.ydata)
            x0, y0 = self.rect_start

            # Fill rectangle in mask
            x_min, x_max = min(x0, x), max(x0, x)
            y_min, y_max = min(y0, y), max(y0, y)

            # Clamp to image bounds
            x_min = max(0, x_min)
            x_max = min(self.width, x_max)
            y_min = max(0, y_min)
            y_max = min(self.height, y_max)

            self.mask[y_min:y_max, x_min:x_max] = 255

            # Remove rectangle patch
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None

            self.rect_start = None
            self.update_overlay()

        self.freehand_points = []

    def draw_brush(self, x, y):
        """Draw a brush stroke at position."""
        # Create circular brush
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        self.mask[dist <= self.brush_size] = 255
        self.update_overlay()

    def on_key(self, event):
        """Handle key press."""
        if event.key == 'r':
            # Reset mask
            self.save_mask_state()
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.update_overlay()
            print("Mask reset")

        elif event.key == 'u':
            # Undo
            if self.mask_history:
                self.mask = self.mask_history.pop()
                self.update_overlay()
                print("Undo")
            else:
                print("Nothing to undo")

        elif event.key == 's' or event.key == 'enter':
            # Save and continue
            self.save_mask()
            self.saved = True
            plt.close(self.fig)

        elif event.key == 'q' or event.key == 'escape':
            # Quit without saving
            print(f"Skipped: {self.image_path.name}")
            plt.close(self.fig)

        elif event.key == 'b':
            # Toggle brush size
            self.brush_idx = (self.brush_idx + 1) % len(self.brush_sizes)
            self.brush_size = self.brush_sizes[self.brush_idx]
            self.update_title()
            print(f"Brush size: {self.brush_size}px")

        elif event.key == 'm':
            # Toggle mode
            self.mode = 'freehand' if self.mode == 'rectangle' else 'rectangle'
            self.update_title()
            print(f"Mode: {self.mode}")

    def save_mask(self):
        """Save the mask to file."""
        output_path = self.image_path.parent / f"{self.image_path.stem}_mask.png"
        mask_img = Image.fromarray(self.mask)
        mask_img.save(output_path)
        print(f"Saved: {output_path}")

    def run(self):
        """Run the interactive mask creator."""
        plt.show()
        return self.saved


def process_directory(input_dir: str):
    """Process all images in a directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory not found: {input_path}")
        return

    extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    images = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in extensions and '_mask' not in f.stem
    ]

    if not images:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(images)} images in {input_path}")
    print("=" * 50)

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")

        # Check if mask already exists
        mask_path = img_path.parent / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            response = input(f"  Mask already exists. Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("  Skipped")
                continue

        creator = InteractiveMaskCreator(str(img_path))
        saved = creator.run()

        if not saved:
            continue_response = input("  Continue to next image? (y/n): ").strip().lower()
            if continue_response != 'y':
                print("Exiting...")
                break

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive mask creation tool for inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Left click + drag  Draw mask region
  r                  Reset current mask
  u                  Undo last region
  s / Enter          Save mask and continue
  q / Escape         Skip current image
  b                  Toggle brush size
  m                  Toggle mode (rectangle/freehand)
        """
    )
    parser.add_argument("--input-dir", type=str, help="Directory containing images")
    parser.add_argument("--image", type=str, help="Single image to process")

    args = parser.parse_args()

    if args.input_dir:
        process_directory(args.input_dir)
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            return

        creator = InteractiveMaskCreator(args.image)
        creator.run()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python interactive_mask.py --input-dir ./test_images/inpainting")
        print("  python interactive_mask.py --image ./test_images/inpainting/photo.jpg")


if __name__ == "__main__":
    main()
