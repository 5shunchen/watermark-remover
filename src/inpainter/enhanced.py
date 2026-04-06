"""
Enhanced AI Inpainting Module
Supports multiple backends: OpenCV, ONNX (LaMa), and external APIs
"""

from typing import Literal, Optional
import cv2
import numpy as np
from PIL import Image

InpaintMethod = Literal["telea", "ns", "ns_original", "lama", "aot"]


class EnhancedInpainter:
    """
    Enhanced inpainter with support for AI-based models
    """

    def __init__(
        self,
        method: str = "ns",
        device: str = "cpu",
    ):
        """
        Initialize the enhanced inpainting module

        Args:
            method: Inpainting method
                - "telea": OpenCV Telea, fast
                - "ns": OpenCV Navier-Stokes, better quality
                - "ns_original": Original NS implementation
                - "lama": LaMa AI model (best quality, requires model)
            device: Device to run on ("cpu" or "cuda")
        """
        self.method = method
        self.device = device
        self.model = None

        # Load AI model if requested
        if method in ["lama", "aot"]:
            self.model = self._try_load_ai_model()

    def _try_load_ai_model(self):
        """Try to load AI inpainting model"""
        try:
            # Try to import ONNX runtime
            import onnxruntime as ort

            # Check for model file
            import os
            model_paths = [
                "models/lama.onnx",
                "models/big-lama.onnx",
                "models/laema.onnx",
            ]

            for model_path in model_paths:
                if os.path.exists(model_path):
                    session_options = ort.SessionOptions()
                    if self.device == "cuda":
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]

                    session = ort.InferenceSession(
                        model_path, sess_options=session_options, providers=providers
                    )
                    print(f"Loaded AI model: {model_path}")
                    return session

            print("No AI model found, falling back to OpenCV methods")
            return None

        except ImportError:
            print("ONNX runtime not available, using OpenCV methods")
            return None
        except Exception as e:
            print(f"Error loading AI model: {e}, falling back to OpenCV")
            return None

    def preprocess_mask(self, mask: Image.Image, target_size: tuple) -> np.ndarray:
        """
        Preprocess the mask for inpainting

        Args:
            mask: Input mask as PIL Image
            target_size: Target size as (width, height)

        Returns:
            Preprocessed mask as numpy array (uint8, 0 or 255)
        """
        # Resize mask to match target image size
        mask_resized = mask.resize(target_size, resample=Image.LANCZOS)

        # Convert to numpy array
        mask_array = np.array(mask_resized)

        # Ensure mask is binary (0 or 255)
        mask_array = (mask_array > 127).astype(np.uint8) * 255

        # Apply morphological dilation to ensure full coverage
        kernel = np.ones((5, 5), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=2)

        return mask_array

    def _inpaint_opencv(self, image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """OpenCV-based inpainting"""
        if self.method == "ns":
            flags = cv2.INPAINT_NS
            radius = 5
        elif self.method == "ns_original":
            flags = cv2.INPAINT_NS
            radius = 3
        else:  # telea
            flags = cv2.INPAINT_TELEA
            radius = 3

        return cv2.inpaint(image_array, mask_array, inpaintRadius=radius, flags=flags)

    def _inpaint_lama(self, image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """LaMa AI model inpainting"""
        if self.model is None:
            return self._inpaint_opencv(image_array, mask_array)

        try:
            # Normalize image to [0, 1]
            img_norm = image_array.astype(np.float32) / 255.0

            # Create binary mask (0 for valid, 1 for masked)
            mask_binary = (mask_array > 127).astype(np.float32)

            # Expand dimensions for batch processing
            img_input = np.expand_dims(img_norm, axis=0)
            mask_input = np.expand_dims(mask_binary, axis=0)

            # Run inference
            result = self.model.run(
                None,
                {"image": img_input, "mask": mask_input}
            )[0]

            # Convert back to uint8
            result_image = (np.clip(result[0], 0, 1) * 255).astype(np.uint8)

            return result_image

        except Exception as e:
            print(f"AI inpainting failed: {e}, falling back to OpenCV")
            return self._inpaint_opencv(image_array, mask_array)

    def _inpaint_multi_pass(self, image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """
        Multi-pass inpainting for better results
        Strategy:
        1. First pass: NS algorithm for structure
        2. Second pass: Telea for refinement
        3. Third pass: Color correction using surrounding pixels
        """
        # First pass with NS (larger radius for structure)
        result_ns = cv2.inpaint(image_array, mask_array, inpaintRadius=7, flags=cv2.INPAINT_NS)

        # Create a refined mask (slightly smaller)
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.erode(mask_array, kernel, iterations=1)

        # Second pass with Telea on the refined mask
        result_refined = cv2.inpaint(result_ns, refined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Third pass: Color correction using bilateral filter for smooth blending
        # Apply bilateral filter to the inpainted region for better edge preservation
        bilateral = cv2.bilateralFilter(result_refined, 9, 75, 75)

        # Blend the bilateral result with the original using the mask
        mask_3ch = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)
        mask_norm = mask_3ch.astype(np.float32) / 255.0

        # Smooth the mask edges for better blending
        mask_smooth = cv2.GaussianBlur(mask_norm, (5, 5), 0)

        # Final blend
        result_final = (result_refined * (1 - mask_smooth) + bilateral * mask_smooth).astype(np.uint8)

        return result_final

    def _inpaint_aggressive(self, image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """
        Aggressive inpainting for difficult watermarks
        Uses larger radius and multiple iterations
        """
        # Dilate mask to cover more area around the watermark
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_array, kernel, iterations=2)

        # First pass: NS with large radius
        result1 = cv2.inpaint(image_array, dilated_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

        # Second pass: Telea on original mask
        result2 = cv2.inpaint(result1, mask_array, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Third pass: Another NS pass
        result3 = cv2.inpaint(result2, mask_array, inpaintRadius=3, flags=cv2.INPAINT_NS)

        return result3

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Perform inpainting to remove watermarks

        Args:
            image: Input PIL Image with watermarks
            mask: Mask indicating watermark areas

        Returns:
            PIL Image with watermarks removed
        """
        # Convert PIL images to numpy arrays
        image_array = np.array(image)

        # Ensure image is in RGB format
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Process the mask
        processed_mask = self.preprocess_mask(mask, image.size)

        # Verify dimensions match
        if image_array.shape[:2] != processed_mask.shape:
            processed_mask = cv2.resize(
                processed_mask, (image_array.shape[1], image_array.shape[0])
            )

        # Apply inpainting
        try:
            if self.method in ["lama", "aot"] and self.model is not None:
                result = self._inpaint_lama(image_array, processed_mask)
            elif self.method == "multi":
                result = self._inpaint_multi_pass(image_array, processed_mask)
            elif self.method == "aggressive":
                result = self._inpaint_aggressive(image_array, processed_mask)
            else:
                result = self._inpaint_opencv(image_array, processed_mask)

            # Convert result back to PIL Image
            return Image.fromarray(result)

        except Exception as e:
            print(f"Inpainting failed with error: {e}")
            # Fallback to simple blur
            return self._fallback_inpaint(image_array, processed_mask)

    def _fallback_inpaint(self, image_array: np.ndarray, mask_array: np.ndarray) -> Image.Image:
        """Fallback inpainting using Gaussian blur"""
        # Apply Gaussian blur to the masked area
        blurred = cv2.GaussianBlur(image_array, (21, 21), 0)

        # Blend using the mask
        mask_3ch = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)
        mask_norm = mask_3ch.astype(np.float32) / 255.0

        result = (image_array * (1 - mask_norm) + blurred * mask_norm).astype(np.uint8)
        return Image.fromarray(result)


# Convenience function
def remove_watermark(
    image: Image.Image,
    mask: Image.Image,
    method: str = "ns",
    device: str = "cpu",
) -> Image.Image:
    """
    Remove watermark from an image using enhanced inpainting

    Args:
        image: Input PIL Image with watermarks
        mask: Mask indicating watermark areas
        method: Inpainting method ("telea", "ns", "ns_original", "lama", "multi")
        device: Device to run on ("cpu" or "cuda")

    Returns:
        PIL Image with watermarks removed
    """
    inpainter = EnhancedInpainter(method=method, device=device)
    return inpainter.inpaint(image, mask)
