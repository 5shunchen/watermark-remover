"""
AI Inpainting Module
Uses OpenCV inpainting algorithms with multiple method support
"""

from typing import Literal, Optional

import cv2
import numpy as np
from PIL import Image

InpaintMethod = Literal["telea", "ns", "ns_original"]


class Inpainter:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        method: str = "telea",
    ):
        """
        Initialize the inpainting module

        Args:
            model_path: Path to the inpainting model (reserved for future AI model integration)
            device: Device to run the model on ("cpu" or "cuda") - reserved for future use
            method: Inpainting method - "telea" (fast), "ns" (high quality), or "ns_original"
        """
        self.device = device
        self.method = method
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """
        Load the inpainting model

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object
        """
        # Placeholder for model loading
        # In a real implementation, this would load an actual inpainting model
        print(f"Loading inpainting model (placeholder) on {self.device}")
        return {"loaded": True, "model_path": model_path}

    def preprocess_mask(self, mask: Image.Image, target_size: tuple) -> np.ndarray:
        """
        Preprocess the mask for inpainting

        Args:
            mask: Input mask as PIL Image
            target_size: Target size as (width, height)

        Returns:
            Preprocessed mask as numpy array
        """
        # Resize mask to match target image size
        mask_resized = mask.resize(target_size, resample=Image.LANCZOS)

        # Convert to numpy array
        mask_array = np.array(mask_resized)

        # Ensure mask is binary (0 or 255)
        mask_array = (mask_array > 127).astype(np.uint8) * 255

        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)

        return mask_array

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

        # Apply inpainting using OpenCV's inpainting algorithms
        try:
            # Select inpainting method based on configuration
            if self.method == "ns":
                # Navier-Stokes - higher quality but slower
                flags = cv2.INPAINT_NS
                radius = 5
            elif self.method == "ns_original":
                # Navier-Stokes original implementation
                flags = cv2.INPAINT_NS
                radius = 3
            else:
                # Telea - fast and good quality (default)
                flags = cv2.INPAINT_TELEA
                radius = 3

            result = cv2.inpaint(
                image_array, processed_mask, inpaintRadius=radius, flags=flags
            )

            # Convert result back to PIL Image
            result_image = Image.fromarray(result)

            return result_image
        except Exception as e:
            print(f"Inpainting failed with error: {e}")
            print("Using backup inpainting method...")

            # Backup method: use median blur to fill masked areas
            mask_binary = (processed_mask > 127).astype(np.uint8)

            # Dilate mask slightly to cover more area
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask_binary, kernel, iterations=1)

            # Apply median blur to the masked area
            blurred = cv2.medianBlur(image_array, 21)

            # Blend the original image with the blurred version using the mask
            result = image_array.copy()
            result[dilated_mask == 1] = blurred[dilated_mask == 1]

            return Image.fromarray(result)


def remove_watermark(
    image: Image.Image,
    mask: Image.Image,
    model_path: Optional[str] = None,
    device: str = "cpu",
    method: str = "telea",
) -> Image.Image:
    """
    Remove watermark from an image using inpainting

    Args:
        image: Input PIL Image with watermarks
        mask: Mask indicating watermark areas
        model_path: Path to the inpainting model (reserved for future use)
        device: Device to run the model on (reserved for future use)
        method: Inpainting method - "telea" (fast), "ns" (high quality), "ns_original"

    Returns:
        PIL Image with watermarks removed
    """
    inpainter = Inpainter(model_path=model_path, device=device, method=method)
    return inpainter.inpaint(image, mask)
