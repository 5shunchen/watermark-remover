"""
PyTorch Hub-based LaMa Inpainting
使用 PyTorch Hub 加载预训练 LaMa 模型
"""

import torch
import cv2
import numpy as np
from PIL import Image


class LaMaInpainter:
    """LaMa inpainting using PyTorch Hub"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"Loading LaMa model on {device}...")

        try:
            # Try to load from PyTorch Hub
            # Note: This requires internet connection for first load
            self.model = torch.hub.load(
                "advimman/lama",
                "BigLama",
                pretrained=True,
                trust_repo=True
            )
            self.model.to(device)
            self.model.eval()
            print("LaMa model loaded successfully!")
            self.available = True
        except Exception as e:
            print(f"Failed to load LaMa model: {e}")
            print("Falling back to OpenCV inpainting")
            self.available = False
            self.model = None

    def preprocess(self, image: Image.Image, mask: Image.Image) -> tuple:
        """Preprocess image and mask for LaMa"""
        # Convert to numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0

        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Resize mask to match image
        if mask_array.shape != img_array.shape[:2]:
            mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))

        # Convert to tensor (HWC -> CHW)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)

        return img_tensor.to(self.device), mask_tensor.to(self.device)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Perform LaMa inpainting"""
        if not self.available:
            # Fallback to OpenCV
            from inpainter.enhanced import remove_watermark
            return remove_watermark(image, mask, method="ns")

        with torch.no_grad():
            img_tensor, mask_tensor = self.preprocess(image, mask)

            # LaMa expects masked image
            masked_img = img_tensor * (1 - mask_tensor)

            # Run inference
            result = self.model(masked_img, mask_tensor)

            # Convert back to numpy
            result_np = result.squeeze().permute(1, 2, 0).cpu().numpy()
            result_uint8 = (np.clip(result_np, 0, 1) * 255).astype(np.uint8)

            return Image.fromarray(result_uint8)


def remove_watermark_lama(
    image: Image.Image,
    mask: Image.Image,
    device: str = "cpu",
) -> Image.Image:
    """
    Remove watermark using LaMa inpainting

    Args:
        image: Input PIL Image with watermarks
        mask: Mask indicating watermark areas
        device: Device to run on ("cpu" or "cuda")

    Returns:
        PIL Image with watermarks removed
    """
    inpainter = LaMaInpainter(device=device)
    return inpainter.inpaint(image, mask)


if __name__ == "__main__":
    # Test
    from pathlib import Path
    from detector import detect_watermark

    test_image = Image.open("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    test_mask = detect_watermark(test_image, method="text")

    result = remove_watermark_lama(test_image, test_mask, device="cpu")
    result.save("output/test_lama_result.png")
    print("Test result saved to output/test_lama_result.png")
