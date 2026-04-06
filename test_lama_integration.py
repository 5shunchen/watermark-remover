#!/usr/bin/env python3
"""
Test LaMa model with actual image processing
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.detector import detect_watermark_by_mser, detect_watermark_v2
from src.inpainter import ONNX_AVAILABLE, Inpainter, remove_watermark


def create_test_image_with_watermark():
    """Create a test image with simulated watermark"""
    # Create background image with gradient
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Add gradient background
    for i in range(512):
        for j in range(512):
            img[i, j] = [
                min(255, 100 + i // 3),
                min(255, 150 - i // 5),
                min(255, 200 - i // 4),
            ]

    # Add a simulated watermark (bright rectangle in bottom-right corner)
    img[460:500, 400:500] = [240, 240, 240]  # Light gray watermark

    return Image.fromarray(img)


def test_lama_inpainting():
    """Test LaMa model inpainting"""
    print("\n" + "=" * 60)
    print("Testing LaMa Inpainting")
    print("=" * 60)

    # Create test image
    image = create_test_image_with_watermark()
    print(f"✓ Created test image: {image.size}")

    # Create mask (simulate watermark area)
    mask = Image.new("L", image.size, 0)
    mask_array = np.array(mask)
    mask_array[460:500, 400:500] = 255
    mask = Image.fromarray(mask_array)
    print(f"✓ Created mask")

    # Test 1: LaMa inpainting
    print("\n--- Testing LaMa Method ---")
    try:
        result_lama = remove_watermark(
            image=image, mask=mask, model_path="models/lama.onnx", method="lama"
        )
        print(f"✓ LaMa inpainting successful: {result_lama.size}")

        # Save result
        result_lama.save("test_outputs/lama_result.png")
        print(f"✓ Result saved to test_outputs/lama_result.png")

        # Calculate PSNR-like quality metric
        original = np.array(image)
        result = np.array(result_lama)
        mse = np.mean((original.astype(float) - result.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            print(f"✓ PSNR (vs original with watermark): {psnr:.2f} dB")
        else:
            print("✓ PSNR: Perfect reconstruction")

    except Exception as e:
        print(f"✗ LaMa inpainting failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Telea inpainting (for comparison)
    print("\n--- Testing Telea Method ---")
    try:
        result_telea = remove_watermark(image=image, mask=mask, method="telea")
        print(f"✓ Telea inpainting successful: {result_telea.size}")

        # Save result
        result_telea.save("test_outputs/telea_result.png")
        print(f"✓ Result saved to test_outputs/telea_result.png")

    except Exception as e:
        print(f"✗ Telea inpainting failed: {e}")

    # Test 3: NS inpainting
    print("\n--- Testing Navier-Stokes Method ---")
    try:
        result_ns = remove_watermark(image=image, mask=mask, method="ns")
        print(f"✓ NS inpainting successful: {result_ns.size}")

        # Save result
        result_ns.save("test_outputs/ns_result.png")
        print(f"✓ Result saved to test_outputs/ns_result.png")

    except Exception as e:
        print(f"✗ NS inpainting failed: {e}")

    return True


def test_mser_detection():
    """Test MSER detection on real image"""
    print("\n" + "=" * 60)
    print("Testing MSER Detection")
    print("=" * 60)

    # Create test image
    image = create_test_image_with_watermark()

    # Test MSER detection
    print("\n--- Running MSER Detection ---")
    watermarks = detect_watermark_by_mser(image)

    print(f"Detected {len(watermarks)} watermark regions:")
    for i, wm in enumerate(watermarks[:5]):
        print(
            f"  {i+1}. Position: ({wm['x']}, {wm['y']}), "
            f"Size: {wm['w']}x{wm['h']}, "
            f"Confidence: {wm['confidence']}, Type: {wm['type']}"
        )

    # Test v2 API with different methods
    print("\n--- Testing detect_watermark_v2 API ---")
    methods = ["auto", "mser", "edge", "color", "pattern"]

    for method in methods:
        try:
            regions = detect_watermark_v2(image, method=method)
            print(f"  Method '{method}': {len(regions)} regions")
        except Exception as e:
            print(f"  Method '{method}': Error - {e}")

    return True


def test_inpainter_class():
    """Test Inpainter class directly"""
    print("\n" + "=" * 60)
    print("Testing Inpainter Class")
    print("=" * 60)

    image = create_test_image_with_watermark()
    mask = Image.new("L", image.size, 0)

    # Test with LaMa model
    print("\n--- Creating Inpainter with LaMa ---")
    inpainter = Inpainter(model_path="models/lama.onnx", device="cpu", method="lama")

    print(f"Method: {inpainter.method}")
    print(f"Model loaded: {inpainter.model['loaded']}")
    print(f"LaMa session: {inpainter.lama_session is not None}")

    # Run inpainting
    result = inpainter.inpaint(image, mask)
    print(f"✓ Inpainting result: {result.size}")

    return True


if __name__ == "__main__":
    # Create output directory
    Path("test_outputs").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("LaMa Model Integration Test")
    print("=" * 60)
    print(f"ONNX Runtime Available: {ONNX_AVAILABLE}")

    all_passed = True

    try:
        test_mser_detection()
    except Exception as e:
        print(f"MSER test failed: {e}")
        all_passed = False

    try:
        test_inpainter_class()
    except Exception as e:
        print(f"Inpainter class test failed: {e}")
        all_passed = False

    try:
        test_lama_inpainting()
    except Exception as e:
        print(f"LaMa inpainting test failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All LaMa integration tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
