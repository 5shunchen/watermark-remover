"""
Basic test to verify the watermark removal functionality works
"""
import numpy as np
from PIL import Image
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import detect_watermark
from inpainter import remove_watermark


def create_test_image():
    """Create a simple test image with a simulated watermark"""
    # Create a 200x200 RGB image with a gradient background
    img_array = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add a gradient background
    for i in range(200):
        for j in range(200):
            img_array[i, j] = [(i * 128 // 200) % 256, (j * 128 // 200) % 256, 100]

    # Add a simulated watermark in the bottom right corner (red rectangle)
    watermark_area = img_array[150:190, 150:190]
    watermark_area[:, :] = [255, 0, 0]  # Red watermark

    return Image.fromarray(img_array)


def test_detection_and_removal():
    """Test the complete watermark detection and removal pipeline"""
    print("Creating test image...")
    test_img = create_test_image()

    print("Detecting watermark...")
    mask = detect_watermark(test_img, method="auto")

    print(f"Detected mask size: {mask.size}")
    print(f"Mask mode: {mask.mode}")

    # Verify that the mask detected something
    mask_array = np.array(mask)
    detected_pixels = np.sum(mask_array > 127)
    print(f"Pixels detected as watermark: {detected_pixels}")

    # At least some pixels should be detected as watermark
    assert detected_pixels > 0, "No watermark detected!"

    print("Removing watermark...")
    cleaned_img = remove_watermark(test_img, mask)

    print(f"Cleaned image size: {cleaned_img.size}")
    print(f"Cleaned image mode: {cleaned_img.mode}")

    # Verify the cleaned image has the correct size
    assert cleaned_img.size == test_img.size

    print("✅ All tests passed!")


def test_imports():
    """Test that all modules can be imported without errors"""
    try:
        from detector import detect_watermark
        from inpainter import remove_watermark
        # Try importing video separately since it has optional dependencies
        try:
            from video import extract_frames
            print("✅ Video module imported successfully!")
        except ImportError as e:
            print(f"⚠️  Video module import failed (this is OK if optional dependencies aren't met): {e}")

        from api.main import app
        print("✅ Other modules imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


if __name__ == "__main__":
    print("Testing imports...")
    test_imports()

    print("\nTesting detection and removal...")
    test_detection_and_removal()

    print("\n🎉 All tests completed successfully!")