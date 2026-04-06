#!/usr/bin/env python3
"""
Test script for new detector and inpainter features:
- MSER watermark detection
- detect_watermark_v2 API
- LaMa model support (if model available)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import (_mask_to_regions, _nms_regions,
                          detect_watermark_by_mser, detect_watermark_by_onnx,
                          detect_watermark_v2)
from src.inpainter import ONNX_AVAILABLE, Inpainter, remove_watermark


def create_test_image():
    """Create a simple test image with simulated watermark"""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Add some content
    img[100:200, 100:200] = [100, 150, 200]  # Blue square

    # Add simulated watermark (bright text-like region in corner)
    img[350:380, 300:380] = [250, 250, 250]  # Bright "watermark"

    return Image.fromarray(img)


def test_mser_detection():
    """Test MSER watermark detection"""
    print("\n=== Testing MSER Detection ===")

    image = create_test_image()
    regions = detect_watermark_by_mser(image)

    print(f"Detected {len(regions)} regions")

    if len(regions) > 0:
        for i, region in enumerate(regions[:3]):  # Show top 3
            print(
                f"  Region {i+1}: x={region['x']}, y={region['y']}, "
                f"w={region['w']}, h={region['h']}, "
                f"confidence={region['confidence']}, type={region['type']}"
            )

    # Verify output format
    assert isinstance(regions, list), "Should return a list"
    if len(regions) > 0:
        region = regions[0]
        assert "x" in region and "y" in region, "Region should have x, y coordinates"
        assert "w" in region and "h" in region, "Region should have w, h dimensions"
        assert "confidence" in region, "Region should have confidence"
        assert "type" in region, "Region should have type"

    print("  ✓ MSER detection test passed")
    return True


def test_detect_watermark_v2():
    """Test the new detect_watermark_v2 API"""
    print("\n=== Testing detect_watermark_v2 API ===")

    image = create_test_image()

    # Test different methods
    methods = ["auto", "mser", "edge", "color"]

    for method in methods:
        regions = detect_watermark_v2(image, method=method)
        print(f"  Method '{method}': {len(regions)} regions detected")

        # Verify output format
        assert isinstance(regions, list), f"Method {method} should return a list"

        if len(regions) > 0:
            region = regions[0]
            assert all(
                key in region for key in ["x", "y", "w", "h", "confidence", "type"]
            ), f"Method {method} should return complete region dict"

    print("  ✓ detect_watermark_v2 API test passed")
    return True


def test_mask_to_regions():
    """Test conversion of mask to regions"""
    print("\n=== Testing Mask to Regions Conversion ===")

    # Create a simple binary mask with two "watermark" regions
    mask_array = np.zeros((200, 200), dtype=np.uint8)
    mask_array[10:30, 150:190] = 255  # Top-right region
    mask_array[170:190, 10:50] = 255  # Bottom-left region

    mask = Image.fromarray(mask_array)
    regions = _mask_to_regions(mask)

    print(f"  Converted mask to {len(regions)} regions")

    assert len(regions) >= 1, "Should detect at least one region"
    assert len(regions) <= 3, "Should not detect too many false regions"

    print("  ✓ Mask to regions conversion test passed")
    return True


def test_nms_regions():
    """Test non-maximum suppression for regions"""
    print("\n=== Testing NMS for Regions ===")

    # Create overlapping regions (high IoU)
    regions = [
        {"x": 100, "y": 100, "w": 50, "h": 50, "confidence": 0.9, "type": "test"},
        {
            "x": 105,
            "y": 105,
            "w": 50,
            "h": 50,
            "confidence": 0.7,
            "type": "test",
        },  # Highly overlapping
        {
            "x": 300,
            "y": 300,
            "w": 50,
            "h": 50,
            "confidence": 0.8,
            "type": "test",
        },  # Non-overlapping
    ]

    filtered = _nms_regions(regions, iou_threshold=0.3)

    print(f"  Reduced {len(regions)} regions to {len(filtered)} after NMS")

    # Should have removed the overlapping region with lower confidence
    # Expected: region with 0.9 confidence and region with 0.8 confidence (non-overlapping)
    assert len(filtered) == 2, f"NMS should reduce to 2 regions, got {len(filtered)}"

    # Highest confidence region should be kept
    assert filtered[0]["confidence"] == 0.9, "NMS should keep highest confidence region"

    # Non-overlapping region should also be kept
    confidences = [r["confidence"] for r in filtered]
    assert 0.8 in confidences, "NMS should keep non-overlapping region"

    print("  ✓ NMS regions test passed")
    return True


def test_inpainter_lama_fallback():
    """Test Inpainter gracefully falls back when LaMa model not available"""
    print("\n=== Testing Inpainter LaMa Fallback ===")

    image = create_test_image()
    mask = Image.new("L", image.size, 0)

    # Test with non-existent model path - should fallback to Telea
    inpainter = Inpainter(model_path="/nonexistent/path/lama.onnx", method="lama")

    # Should have fallen back to Telea
    assert inpainter.method == "telea", "Should fallback to Telea when model not found"

    # Should still produce output
    result = inpainter.inpaint(image, mask)
    assert result.size == image.size, "Output should match input size"

    print("  ✓ Inpainter LaMa fallback test passed")
    return True


def test_inpainter_methods():
    """Test different inpainting methods"""
    print("\n=== Testing Inpainter Methods ===")

    image = create_test_image()
    mask = Image.new("L", image.size, 0)

    # Add a small "watermark" area to the mask
    mask_array = np.zeros((400, 400), dtype=np.uint8)
    mask_array[350:380, 300:380] = 255
    mask = Image.fromarray(mask_array)

    methods = ["telea", "ns", "ns_original"]

    for method in methods:
        inpainter = Inpainter(method=method)
        result = inpainter.inpaint(image, mask)

        assert result.size == image.size, f"Method {method} output size mismatch"
        assert result.mode == "RGB", f"Method {method} output mode should be RGB"
        print(f"  Method '{method}': OK")

    print("  ✓ All inpainter methods test passed")
    return True


def test_onnx_available():
    """Test ONNX runtime availability"""
    print("\n=== Testing ONNX Runtime Availability ===")
    print(f"  ONNX_AVAILABLE = {ONNX_AVAILABLE}")

    if ONNX_AVAILABLE:
        print("  ✓ ONNX runtime is available")
    else:
        print("  ⚠ ONNX runtime not available (install with: pip install onnxruntime)")

    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing New Detector and Inpainter Features")
    print("=" * 60)

    tests = [
        ("ONNX Runtime Check", test_onnx_available),
        ("MSER Detection", test_mser_detection),
        ("detect_watermark_v2 API", test_detect_watermark_v2),
        ("Mask to Regions", test_mask_to_regions),
        ("NMS Regions", test_nms_regions),
        ("Inpainter LaMa Fallback", test_inpainter_lama_fallback),
        ("Inpainter Methods", test_inpainter_methods),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test_name} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
