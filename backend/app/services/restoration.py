"""
Image restoration service.

Handles:
- Face enhancement (CodeFormer, GFPGAN)
- Image upscaling (Real-ESRGAN)
- Scratch removal
- Colorization

These models are separate from diffusers and have their own dependencies.
"""

import os
import base64
import io
import time
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

# Lazy imports - these will be loaded only when needed
# to avoid import errors if dependencies aren't installed


class RestorationService:
    """Service for image restoration tasks."""

    def __init__(self):
        """Initialize the service - models are loaded lazily."""
        self.device = "cuda"

        # Cached model instances
        self._codeformer = None
        self._gfpgan = None
        self._realesrgan = None
        self._realesrgan_scale = None

        print("RestorationService initialized")

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
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_codeformer(self, fidelity: float = 0.5):
        """Load CodeFormer model."""
        if self._codeformer is not None:
            return self._codeformer

        print("Loading CodeFormer...")
        try:
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            from codeformer.basicsr.utils import img2tensor, tensor2img
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer
            import torch

            # Download model weights if needed
            from basicsr.utils.download_util import load_file_from_url

            model_path = load_file_from_url(
                url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                model_dir='weights/CodeFormer',
                progress=True,
                file_name='codeformer.pth'
            )

            # Initialize model
            model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)

            checkpoint = torch.load(model_path)['params_ema']
            model.load_state_dict(checkpoint)
            model.eval()

            # Initialize face helper
            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=self.device
            )

            self._codeformer = {
                'model': model,
                'face_helper': face_helper,
                'img2tensor': img2tensor,
                'tensor2img': tensor2img,
            }
            print("CodeFormer loaded successfully")

        except ImportError as e:
            print(f"CodeFormer not available: {e}")
            print("Install with: pip install codeformer-pip")
            raise ImportError(
                "CodeFormer is not installed. Install with: pip install codeformer-pip"
            )

        return self._codeformer

    def _load_gfpgan(self):
        """Load GFPGAN model."""
        if self._gfpgan is not None:
            return self._gfpgan

        print("Loading GFPGAN...")
        try:
            from gfpgan import GFPGANer

            self._gfpgan = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                device=self.device
            )
            print("GFPGAN loaded successfully")

        except ImportError as e:
            print(f"GFPGAN not available: {e}")
            print("Install with: pip install gfpgan")
            raise ImportError(
                "GFPGAN is not installed. Install with: pip install gfpgan"
            )

        return self._gfpgan

    def _load_realesrgan(self, scale: int = 2):
        """Load Real-ESRGAN model."""
        if self._realesrgan is not None and self._realesrgan_scale == scale:
            return self._realesrgan

        print(f"Loading Real-ESRGAN ({scale}x)...")
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Select model based on scale
            if scale == 4:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            else:  # scale == 2
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

            self._realesrgan = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=0,  # 0 for no tile, increase if OOM
                tile_pad=10,
                pre_pad=0,
                half=True,  # Use fp16 for faster inference
                device=self.device
            )
            self._realesrgan_scale = scale
            print(f"Real-ESRGAN ({scale}x) loaded successfully")

        except ImportError as e:
            print(f"Real-ESRGAN not available: {e}")
            print("Install with: pip install realesrgan")
            raise ImportError(
                "Real-ESRGAN is not installed. Install with: pip install realesrgan"
            )

        return self._realesrgan

    # =========================================================================
    # Restoration Methods
    # =========================================================================

    def enhance_faces_codeformer(
        self, image: Image.Image, fidelity: float = 0.5
    ) -> Image.Image:
        """
        Enhance faces using CodeFormer.

        Args:
            image: Input PIL Image
            fidelity: Balance between quality (0) and fidelity (1)

        Returns:
            Enhanced PIL Image
        """
        import torch

        codeformer = self._load_codeformer(fidelity)
        model = codeformer['model']
        face_helper = codeformer['face_helper']
        img2tensor = codeformer['img2tensor']
        tensor2img = codeformer['tensor2img']

        # Convert to cv2 format
        img = self.pil_to_cv2(image)

        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False)
        face_helper.align_warp_face()

        # Process each detected face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = model(cropped_face_t, w=fidelity, adain=True)[0]
                    restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                del output
            except RuntimeError as e:
                print(f'CodeFormer failed on face {idx}: {e}')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()

        return self.cv2_to_pil(restored_img)

    def enhance_faces_gfpgan(self, image: Image.Image) -> Image.Image:
        """
        Enhance faces using GFPGAN.

        Args:
            image: Input PIL Image

        Returns:
            Enhanced PIL Image
        """
        gfpgan = self._load_gfpgan()

        # Convert to cv2 format
        img = self.pil_to_cv2(image)

        # Enhance
        _, _, restored_img = gfpgan.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        return self.cv2_to_pil(restored_img)

    def upscale(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.

        Args:
            image: Input PIL Image
            scale: Upscaling factor (2 or 4)

        Returns:
            Upscaled PIL Image
        """
        realesrgan = self._load_realesrgan(scale)

        # Convert to cv2 format
        img = self.pil_to_cv2(image)

        # Upscale
        output, _ = realesrgan.enhance(img, outscale=scale)

        return self.cv2_to_pil(output)

    def remove_scratches(self, image: Image.Image) -> Image.Image:
        """
        Remove scratches and artifacts.

        This uses a simple denoising approach. For better results,
        consider using a dedicated model like Bringing-Old-Photos-Back-to-Life.

        Args:
            image: Input PIL Image

        Returns:
            Processed PIL Image
        """
        # Convert to cv2 format
        img = self.pil_to_cv2(image)

        # Apply non-local means denoising
        # This helps with minor scratches and noise
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        return self.cv2_to_pil(denoised)

    def colorize(self, image: Image.Image) -> Image.Image:
        """
        Colorize a black and white image.

        TODO: Implement proper colorization using DeOldify or similar.
        Currently returns the original image.

        Args:
            image: Input PIL Image (grayscale or color)

        Returns:
            Colorized PIL Image
        """
        # Placeholder - proper implementation would use DeOldify
        print("Colorization not yet implemented - returning original image")
        return image

    # =========================================================================
    # Main Restoration Pipeline
    # =========================================================================

    def restore(
        self,
        image_b64: str,
        enable_face_enhance: bool = True,
        face_model: str = "codeformer",
        fidelity: float = 0.5,
        upscale: str = "2x",
        enable_scratch_removal: bool = False,
        enable_colorize: bool = False,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Full restoration pipeline.

        Args:
            image_b64: Base64 encoded input image
            enable_face_enhance: Enable face enhancement
            face_model: 'codeformer' or 'gfpgan'
            fidelity: CodeFormer fidelity (0-1)
            upscale: 'none', '2x', or '4x'
            enable_scratch_removal: Enable scratch removal
            enable_colorize: Enable colorization

        Returns:
            Tuple of (result_b64, error_message, processing_time)
        """
        start_time = time.time()

        try:
            # Decode input (also captures original format)
            image, original_format = self.base64_to_image(image_b64)
            result = image

            # Step 1: Scratch/Artifact Removal (do first to clean image)
            if enable_scratch_removal:
                print("Removing scratches...")
                result = self.remove_scratches(result)

            # Step 2: Face Enhancement
            if enable_face_enhance:
                print(f"Enhancing faces with {face_model}...")
                if face_model == "codeformer":
                    result = self.enhance_faces_codeformer(result, fidelity)
                else:  # gfpgan
                    result = self.enhance_faces_gfpgan(result)

            # Step 3: Colorization
            if enable_colorize:
                print("Colorizing...")
                result = self.colorize(result)

            # Step 4: Upscaling (do last to preserve quality)
            if upscale != "none":
                scale_factor = 4 if upscale == "4x" else 2
                print(f"Upscaling {scale_factor}x...")
                result = self.upscale(result, scale_factor)

            # Encode result in original format
            result_b64 = self.image_to_base64(result, original_format)
            processing_time = time.time() - start_time

            return result_b64, None, processing_time

        except ImportError as e:
            return None, str(e), time.time() - start_time
        except Exception as e:
            return None, f"Restoration failed: {str(e)}", time.time() - start_time

    def unload_models(self):
        """Unload all models to free memory."""
        import torch

        self._codeformer = None
        self._gfpgan = None
        self._realesrgan = None
        self._realesrgan_scale = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Restoration models unloaded")


# =============================================================================
# Singleton Instance
# =============================================================================

_restoration_service: Optional[RestorationService] = None


def get_restoration_service() -> RestorationService:
    """Get or create the RestorationService singleton."""
    global _restoration_service
    if _restoration_service is None:
        _restoration_service = RestorationService()
    return _restoration_service
