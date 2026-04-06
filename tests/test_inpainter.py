"""
Tests for inpainting module
"""

import numpy as np
import pytest
from PIL import Image

from src.inpainter import Inpainter, remove_watermark


def create_test_image_with_watermark(width=100, height=100):
    """Create a test image with a simulated watermark"""
    # Create a base image with some content
    img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

    # Add a simulated watermark in the center
    watermark_start_x, watermark_end_x = width // 3, 2 * width // 3
    watermark_start_y, watermark_end_y = height // 3, 2 * height // 3

    # Make the watermark region slightly different color
    img_array[
        watermark_start_y:watermark_end_y, watermark_start_x:watermark_end_x, :
    ] = [np.clip(c + 30, 0, 255) for c in [100, 150, 200]]

    return Image.fromarray(img_array)


def create_test_mask(width=100, height=100):
    """Create a test mask for the watermark area"""
    mask_array = np.zeros((height, width), dtype=np.uint8)

    # Mark the central area as watermark (value 255)
    watermark_start_x, watermark_end_x = width // 3, 2 * width // 3
    watermark_start_y, watermark_end_y = height // 3, 2 * height // 3

    mask_array[watermark_start_y:watermark_end_y, watermark_start_x:watermark_end_x] = (
        255
    )

    return Image.fromarray(mask_array)


def test_inpainter_initialization():
    """Test that Inpainter initializes correctly"""
    inpainter = Inpainter(device="cpu")

    assert inpainter is not None
    assert inpainter.device == "cpu"
    assert inpainter.model is not None


def test_preprocess_mask():
    """Test mask preprocessing"""
    inpainter = Inpainter(device="cpu")

    # Create a test mask
    original_mask = create_test_mask(100, 100)

    # Preprocess it for a different target size
    target_size = (150, 150)
    processed_mask = inpainter.preprocess_mask(original_mask, target_size)

    # Check dimensions
    assert processed_mask.shape == target_size[::-1]  # Height, Width

    # Check that values are in expected range (0 or 255 after binarization)
    unique_values = np.unique(processed_mask)
    assert all(v in [0, 255] for v in unique_values)


def test_remove_watermark():
    """Test the remove_watermark function"""
    # Create test image and mask
    test_image = create_test_image_with_watermark(100, 100)
    test_mask = create_test_mask(100, 100)

    # Test the remove_watermark function
    result = remove_watermark(test_image, test_mask, device="cpu")

    # Check that result is a valid image
    assert isinstance(result, Image.Image)
    assert result.size == test_image.size
    assert result.mode in ["RGB", "RGBA", "L"]


def test_inpainter_inpaint():
    """Test the inpaint method of Inpainter class"""
    inpainter = Inpainter(device="cpu")

    # Create test image and mask
    test_image = create_test_image_with_watermark(100, 100)
    test_mask = create_test_mask(100, 100)

    # Test the inpaint method
    result = inpainter.inpaint(test_image, test_mask)

    # Check that result is a valid image
    assert isinstance(result, Image.Image)
    assert result.size == test_image.size
    assert result.mode in ["RGB", "RGBA", "L"]


def test_different_image_modes():
    """Test inpainting with different image modes"""
    inpainter = Inpainter(device="cpu")

    # Test RGB image
    rgb_image = create_test_image_with_watermark(80, 80)
    rgb_mask = create_test_mask(80, 80)

    result_rgb = inpainter.inpaint(rgb_image, rgb_mask)
    assert result_rgb.size == rgb_image.size

    # Test grayscale image
    gray_image = rgb_image.convert("L")
    gray_mask = rgb_mask  # Mask stays the same

    result_gray = inpainter.inpaint(gray_image, gray_mask)
    assert result_gray.size == gray_image.size
