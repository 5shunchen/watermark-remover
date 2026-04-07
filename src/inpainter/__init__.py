"""
AI Inpainting Module
Uses OpenCV inpainting algorithms with multiple method support
Also supports LaMa (Large Mask) model for high-quality inpainting
"""

import os
from typing import Literal, Optional

import cv2
import numpy as np
from PIL import Image

# Optional ONNX runtime import for LaMa model
try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession

    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    InferenceSession = None
    ONNX_AVAILABLE = False

InpaintMethod = Literal["telea", "ns", "ns_original", "lama"]


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
            model_path: Path to the inpainting model (for LaMa: "models/lama.onnx")
            device: Device to run the model on ("cpu" or "cuda")
            method: Inpainting method - "telea" (fast), "ns" (high quality),
                    "ns_original", or "lama" (AI-based, best quality)
        """
        self.device = device
        self.method = method
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.lama_session: Optional[InferenceSession] = None

        # Auto-detect LaMa model path if method is "lama"
        if method == "lama" and model_path is None:
            # Get the project root directory
            project_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )
            default_paths = [
                os.path.join(project_root, "models", "lama.onnx"),
                "models/lama.onnx",
                os.path.join(os.getcwd(), "models", "lama.onnx"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self.model_path = path
                    print(f"Found LaMa model at {path}")
                    break
            else:
                print(f"LaMa model not found in any of: {default_paths}")

    def _load_model(self, model_path: Optional[str]):
        """
        Load the inpainting model

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object or session
        """
        if self.method == "lama":
            if model_path is None or not os.path.exists(model_path):
                print(f"LaMa model not found at {model_path}, falling back to Telea")
                self.method = "telea"
                return {"loaded": False, "fallback": "telea"}

            if not ONNX_AVAILABLE:
                print("ONNX runtime not available, falling back to Telea")
                print("Install with: pip install onnxruntime")
                self.method = "telea"
                return {"loaded": False, "fallback": "telea"}

            # Load ONNX model for LaMa
            try:
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self.lama_session = ort.InferenceSession(
                    model_path, providers=providers
                )
                print(f"LaMa model loaded successfully from {model_path}")
                return {
                    "loaded": True,
                    "model_path": model_path,
                    "session": self.lama_session,
                }
            except Exception as e:
                print(f"Failed to load LaMa model: {e}, falling back to Telea")
                self.method = "telea"
                return {"loaded": False, "fallback": "telea"}
        else:
            # OpenCV methods don't require model loading
            return {"loaded": True, "method": self.method}

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

        # Apply inpainting using OpenCV's inpainting algorithms or LaMa model
        try:
            # LaMa model inference
            if self.method == "lama" and self.lama_session is not None:
                return self._inpaint_with_lama(image_array, processed_mask)

            # Select OpenCV inpainting method based on configuration
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

    def _inpaint_with_lama(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray,
    ) -> Image.Image:
        """
        Perform inpainting using LaMa model

        LaMa model requires 512x512 input. Images are resized accordingly.

        Args:
            image_array: Image as numpy array (RGB format)
            mask_array: Binary mask array (255 for areas to inpaint)

        Returns:
            PIL Image with watermarks removed
        """
        if self.lama_session is None:
            raise RuntimeError("LaMa model not loaded")

        # Store original dimensions
        orig_h, orig_w = image_array.shape[:2]

        # LaMa requires 512x512 input
        target_size = (512, 512)

        # Resize image and mask to model input size
        img_resized = cv2.resize(
            image_array, target_size, interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_array, target_size, interpolation=cv2.INTER_LINEAR
        )

        # Normalize image to [0, 1] range (RGB format for LaMa)
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Normalize mask to [0, 1] range
        mask_normalized = mask_resized.astype(np.float32) / 255.0

        # Get input/output names from model
        input_names = [inp.name for inp in self.lama_session.get_inputs()]
        output_name = self.lama_session.get_outputs()[0].name

        # Prepare input tensors
        # Image: (batch, channels, height, width) - RGB format
        img_input = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]

        # Mask: (batch, 1, height, width)
        if len(mask_normalized.shape) == 2:
            mask_input = mask_normalized[np.newaxis, np.newaxis, ...]
        else:
            # If mask has channel dimension, take first channel
            mask_input = mask_normalized[:, :, 0:1][np.newaxis, ...]
            mask_input = np.transpose(mask_input, (0, 3, 1, 2))

        # Run inference
        result = self.lama_session.run(
            None,
            {
                input_names[0]: img_input,
                input_names[1]: mask_input,
            },
        )

        # Get output and convert back to image
        output = result[0]

        # Remove batch dimension and transpose back to HWC format
        output = np.transpose(output[0], (1, 2, 0))

        # Clip to valid range and convert to uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        # Resize back to original dimensions
        output_resized = cv2.resize(
            output, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # Ensure RGB format (LaMa output is already RGB)
        if len(output_resized.shape) == 2:
            output_resized = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2RGB)
        elif output_resized.shape[2] == 4:
            output_resized = cv2.cvtColor(output_resized, cv2.COLOR_RGBA2RGB)

        return Image.fromarray(output_resized)


def remove_watermark(
    image: Image.Image,
    mask: Image.Image,
    model_path: Optional[str] = None,
    device: str = "cpu",
    method: str = "lama",
) -> Image.Image:
    """
    Remove watermark from an image using inpainting

    Args:
        image: Input PIL Image with watermarks
        mask: Mask indicating watermark areas (white=watermark, black=background)
        model_path: Path to the LaMa model file (e.g., "models/lama.onnx")
        device: Device to run the model on ("cpu" or "cuda")
        method: Inpainting method - "telea" (fast), "ns" (high quality),
                "ns_original", or "lama" (AI-based, best quality, default)

    Returns:
        PIL Image with watermarks removed

    Example:
        >>> from PIL import Image
        >>> from src.inpainter import remove_watermark
        >>> image = Image.open("photo_with_watermark.jpg")
        >>> mask = Image.open("watermark_mask.png")
        >>> result = remove_watermark(image, mask, method="lama")
        >>> result.save("photo_clean.jpg")
    """
    inpainter = Inpainter(model_path=model_path, device=device, method=method)
    return inpainter.inpaint(image, mask)
