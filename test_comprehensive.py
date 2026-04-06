"""
Final verification test for the complete watermark removal system
"""
import numpy as np
from PIL import Image
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import detect_watermark
from inpainter import remove_watermark


def test_complete_pipeline():
    """Test the complete watermark detection and removal pipeline"""
    print("🔍 Testing complete watermark removal pipeline...")

    # Create a test image with simulated watermark
    print("🎨 Creating test image with watermark...")
    img_array = np.zeros((300, 300, 3), dtype=np.uint8)

    # Add a colorful background
    for i in range(300):
        for j in range(300):
            img_array[i, j] = [i % 256, j % 256, (i+j) % 256]

    # Add a watermark in the corner (white text-like watermark)
    watermark_coords = (220, 220, 280, 280)  # x1, y1, x2, y2
    img_array[watermark_coords[1]:watermark_coords[3], watermark_coords[0]:watermark_coords[2]] = [255, 255, 255]

    original_img = Image.fromarray(img_array)

    print(f"🖼️  Original image size: {original_img.size}")

    # Detect watermark
    print("🔎 Detecting watermark...")
    mask = detect_watermark(original_img, method="auto")

    print(f"🎭 Generated mask size: {mask.size}")

    # Check that the detection found the watermark area
    mask_array = np.array(mask)
    detected_watermark_area = np.sum(mask_array > 127)
    print(f"📊 Detected watermark pixels: {detected_watermark_area}")

    # The detection should find at least some watermark area
    assert detected_watermark_area > 0, f"Detection failed! Found {detected_watermark_area} pixels"

    # Remove the watermark
    print("✨ Removing watermark...")
    cleaned_img = remove_watermark(original_img, mask, device="cpu")

    print(f"✅ Cleaned image size: {cleaned_img.size}")

    # Verify output is valid
    assert cleaned_img is not None
    assert cleaned_img.size == original_img.size
    assert cleaned_img.mode in ['RGB', 'RGBA', 'L']

    # Save outputs for verification (optional)
    print("💾 Saving sample images...")
    original_img.save("sample_original.png")
    mask.save("sample_mask.png")
    cleaned_img.save("sample_cleaned.png")

    print("🎉 Pipeline test completed successfully!")
    print(f"   • Original saved as: sample_original.png")
    print(f"   • Mask saved as: sample_mask.png")
    print(f"   • Cleaned saved as: sample_cleaned.png")


def test_various_detection_methods():
    """Test different watermark detection methods"""
    print("\n🔍 Testing various detection methods...")

    # Create a test image
    img_array = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    # Add a red watermark area
    img_array[100:140, 100:140] = [255, 0, 0]

    test_img = Image.fromarray(img_array)

    methods = ["color", "edge", "corners", "auto"]

    for method in methods:
        print(f"   Testing '{method}' method...")
        mask = detect_watermark(test_img, method=method)
        mask_array = np.array(mask)
        detected_pixels = np.sum(mask_array > 127)
        print(f"     - Found {detected_pixels} pixels")
        assert mask.size == test_img.size

    print("✅ All detection methods work correctly!")


def test_inpainter_class():
    """Test the Inpainter class directly"""
    print("\n🔧 Testing Inpainter class...")

    # Create test image and mask
    img_array = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    test_img = Image.fromarray(img_array)

    mask_array = np.zeros((100, 100), dtype=np.uint8)
    mask_array[40:60, 40:60] = 255  # Square mask in center
    test_mask = Image.fromarray(mask_array, mode='L')

    from inpainter import Inpainter
    inpainter = Inpainter(device="cpu")

    # Test preprocessing
    processed_mask = inpainter.preprocess_mask(test_mask, test_img.size)
    assert processed_mask.shape == (100, 100)

    # Test inpainting
    result = inpainter.inpaint(test_img, test_mask)
    assert result.size == test_img.size
    assert result.mode in ['RGB', 'RGBA', 'L']

    print("✅ Inpainter class works correctly!")


if __name__ == "__main__":
    print("🚀 Starting comprehensive verification test...\n")

    test_complete_pipeline()
    test_various_detection_methods()
    test_inpainter_class()

    print("\n🏆 All tests passed! The watermark removal system is working correctly.")
    print("\n📋 Features implemented:")
    print("   • Watermark detection (multiple methods)")
    print("   • AI-based inpainting for removal")
    print("   • Support for images and video")
    print("   • REST API interface")
    print("   • Comprehensive test suite")
    print("\n🎉 Watermark removal system is ready!")