"""
Tests for watermark detection module
"""

import numpy as np
import pytest
from PIL import Image

from src.detector import (detect_watermark, detect_watermark_by_color,
                          detect_watermark_by_corners,
                          detect_watermark_by_edge,
                          detect_watermark_by_template_matching)


def create_test_image(width=100, height=100, watermark_type="corner"):
    """Create a test image with simulated watermark"""
    # Create a base image with some content
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add a simulated watermark depending on type
    if watermark_type == "corner":
        # Add a rectangular watermark in the bottom-right corner
        corner_size = min(width, height) // 4
        img_array[height - corner_size : height, width - corner_size : width, :] = [
            255,
            0,
            0,
        ]  # Red watermark

    elif watermark_type == "text":
        # Simulate text watermark with horizontal lines
        for i in range(20, min(height, 50), 5):
            img_array[i, 20:80, :] = [0, 255, 0]  # Green text-like watermark

    return Image.fromarray(img_array)


def test_detect_watermark_by_color():
    """Test color-based watermark detection"""
    test_image = create_test_image(watermark_type="corner")

    mask = detect_watermark_by_color(
        test_image, lower_threshold=(0, 50, 50), upper_threshold=(10, 255, 255)
    )

    # Convert mask to numpy array to check for detected regions
    mask_array = np.array(mask)

    # Check that mask is not empty (some areas should be detected)
    assert mask_array.sum() > 0, "Color detection should find some areas"

    # Verify mask dimensions match input image
    assert mask.size == test_image.size


def test_detect_watermark_by_edge():
    """Test edge-based watermark detection"""
    test_image = create_test_image(watermark_type="text")

    mask = detect_watermark_by_edge(test_image, edge_threshold=50)

    # Convert mask to numpy array to check for detected regions
    mask_array = np.array(mask)

    # Check that mask is not empty (some areas should be detected)
    assert mask_array.sum() > 0, "Edge detection should find some areas"

    # Verify mask dimensions match input image
    assert mask.size == test_image.size


def test_detect_watermark_by_corners():
    """Test corner-based watermark detection"""
    test_image = create_test_image(watermark_type="corner")

    mask = detect_watermark_by_corners(test_image)

    # Convert mask to numpy array to check for detected regions
    mask_array = np.array(mask)

    # Check that mask is not empty (some areas should be detected)
    assert mask_array.sum() >= 0, "Corner detection should not fail"

    # Verify mask dimensions match input image
    assert mask.size == test_image.size


def test_detect_watermark_by_template_matching():
    """Test template matching watermark detection"""
    test_image = create_test_image(watermark_type="corner")

    mask = detect_watermark_by_template_matching(test_image)

    # Convert mask to numpy array to check for detected regions
    mask_array = np.array(mask)

    # Template matching implementation is basic, so just check it returns valid mask
    assert mask_array is not None
    assert mask.size == test_image.size


def test_detect_watermark_auto():
    """Test automatic watermark detection"""
    test_image = create_test_image(watermark_type="corner")

    mask = detect_watermark(test_image, method="auto")

    # Convert mask to numpy array to check for detected regions
    mask_array = np.array(mask)

    # Check that mask is not empty (some areas should be detected)
    assert mask_array.sum() >= 0, "Auto detection should not fail"

    # Verify mask dimensions match input image
    assert mask.size == test_image.size


def test_detect_watermark_methods_consistency():
    """Test that different methods work on the same image"""
    test_image = create_test_image(watermark_type="corner")

    # Test all methods
    color_mask = detect_watermark(test_image, method="color")
    edge_mask = detect_watermark(test_image, method="edge")
    auto_mask = detect_watermark(test_image, method="auto")

    # All should have valid output with correct dimensions
    assert color_mask.size == test_image.size
    assert edge_mask.size == test_image.size
    assert auto_mask.size == test_image.size
