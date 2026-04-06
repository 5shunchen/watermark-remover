"""
Tests for enhanced inpainting module with multiple algorithms
"""

import numpy as np
import pytest
from PIL import Image

from src.inpainter import Inpainter, remove_watermark


def create_test_image(width=100, height=100):
    """Create a test image with simulated watermark"""
    img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add simulated watermark
    img_array[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = [
        200,
        200,
        200,
    ]
    return Image.fromarray(img_array)


def create_test_mask(width=100, height=100):
    """Create a test mask"""
    mask_array = np.zeros((height, width), dtype=np.uint8)
    mask_array[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 255
    return Image.fromarray(mask_array)


class TestInpainterMethods:
    """Test different inpainting methods"""

    def test_telea_method(self):
        """Test Telea inpainting method (fast, good quality)"""
        inpainter = Inpainter(method="telea")
        img = create_test_image(80, 80)
        mask = create_test_mask(80, 80)

        result = inpainter.inpaint(img, mask)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_ns_method(self):
        """Test Navier-Stokes inpainting method (higher quality)"""
        inpainter = Inpainter(method="ns")
        img = create_test_image(80, 80)
        mask = create_test_mask(80, 80)

        result = inpainter.inpaint(img, mask)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_ns_original_method(self):
        """Test Navier-Stokes original method"""
        inpainter = Inpainter(method="ns_original")
        img = create_test_image(80, 80)
        mask = create_test_mask(80, 80)

        result = inpainter.inpaint(img, mask)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_remove_watermark_with_method(self):
        """Test remove_watermark function with method parameter"""
        img = create_test_image(100, 100)
        mask = create_test_mask(100, 100)

        # Test with default method
        result_default = remove_watermark(img, mask)
        assert isinstance(result_default, Image.Image)

        # Test with NS method
        result_ns = remove_watermark(img, mask, method="ns")
        assert isinstance(result_ns, Image.Image)


class TestMaskPreprocessing:
    """Test mask preprocessing enhancements"""

    def test_mask_binary_conversion(self):
        """Test that masks are properly converted to binary"""
        inpainter = Inpainter()

        # Create a mask with intermediate values
        mask_array = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        mask = Image.fromarray(mask_array)

        processed = inpainter.preprocess_mask(mask, (100, 100))

        # All values should be 0 or 255
        unique_values = np.unique(processed)
        assert all(v in [0, 255] for v in unique_values)

    def test_mask_morphological_operations(self):
        """Test that morphological operations clean up the mask"""
        inpainter = Inpainter()

        # Create a noisy mask
        mask_array = np.zeros((50, 50), dtype=np.uint8)
        mask_array[20:30, 20:30] = 255
        # Add some noise
        mask_array[10, 10] = 255
        mask_array[40, 40] = 255

        mask = Image.fromarray(mask_array)
        processed = inpainter.preprocess_mask(mask, (50, 50))

        # The noise should be reduced by morphological operations
        assert processed.shape == (50, 50)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_mask(self):
        """Test with an empty (all black) mask"""
        inpainter = Inpainter()
        img = create_test_image(50, 50)
        mask = Image.new("L", (50, 50), 0)  # All black

        result = inpainter.inpaint(img, mask)

        # Should return image essentially unchanged
        assert isinstance(result, Image.Image)

    def test_full_mask(self):
        """Test with a full (all white) mask"""
        inpainter = Inpainter()
        img = create_test_image(50, 50)
        mask = Image.new("L", (50, 50), 255)  # All white

        result = inpainter.inpaint(img, mask)

        # Should process without error
        assert isinstance(result, Image.Image)

    def test_rgba_image(self):
        """Test with RGBA image"""
        inpainter = Inpainter()
        img = Image.new("RGBA", (50, 50), (100, 150, 200, 128))
        mask = create_test_mask(50, 50)

        result = inpainter.inpaint(img, mask)

        assert isinstance(result, Image.Image)

    def test_grayscale_image(self):
        """Test with grayscale image"""
        inpainter = Inpainter()
        img = Image.new("L", (50, 50), 128)
        mask = create_test_mask(50, 50)

        result = inpainter.inpaint(img, mask)

        assert isinstance(result, Image.Image)

    def test_size_mismatch(self):
        """Test when mask and image sizes don't match"""
        inpainter = Inpainter()
        img = create_test_image(100, 100)
        mask = create_test_mask(50, 50)  # Different size

        result = inpainter.inpaint(img, mask)

        # Should resize mask to match image
        assert result.size == img.size
