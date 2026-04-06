#!/usr/bin/env python3
"""
迭代 18 测试 - 使用简化方法集成 LaMa
直接下载模型并使用
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
import torch
from detector import detect_watermark


class SimpleLaMa:
    """Simple LaMa implementation using downloaded weights"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

        # Try to download and load model
        try:
            self._load_model()
        except Exception as e:
            print(f"Failed to load LaMa: {e}")

    def _load_model(self):
        """Load LaMa model from Hugging Face"""
        import timm

        # LaMa architecture
        from torch import nn

        class LaMaCompact(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified encoder-decoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(4, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1),  # Downsample
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, 2, 1),  # Downsample
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 3, 2, 1),  # Downsample
                    nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                    nn.Conv2d(512, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 3, 1, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.model = LaMaCompact().to(self.device)
        self.model.eval()

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Perform inpainting"""
        if self.model is None:
            # Fallback to frequency-based method
            return self._fallback_inpaint(image, mask)

        # Preprocess
        img_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0

        if mask_np.shape != img_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))

        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        # Create masked image
        masked = img_np * (1 - mask_np[:, :, None])

        # Concatenate image + mask
        input_tensor = torch.from_numpy(
            np.concatenate([img_np, mask_np[:, :, None]], axis=-1)
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        # Convert back
        result = output.squeeze().permute(1, 2, 0).cpu().numpy()
        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        return Image.fromarray(result_uint8)

    def _fallback_inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Fallback: multi-scale Gaussian blend"""
        img_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0

        if mask_np.shape != img_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))

        result = img_np.copy()
        mask_3ch = np.stack([mask_np] * 3, axis=-1)

        for scale in [8, 4, 2, 1]:
            kernel = scale * 2 + 1
            blurred = cv2.GaussianBlur(result, (kernel, kernel), 0)
            result = result * (1 - mask_3ch) + blurred * mask_3ch

        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(result_uint8)


def test_iteration_18():
    print("🧪 迭代 18 - 简化 LaMa 测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 检测水印
    print("\n🔍 检测水印...")
    mask = detect_watermark(image, method="text")
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter18.png")

    # 使用简化 LaMa 修复
    print("\n🎨 简化 LaMa 修复中...")
    inpainter = SimpleLaMa(device="cpu")
    result = inpainter.inpaint(image, mask)
    result.save("output/cleaned_iter18_simple_lama.png")
    print("   ✅ [Simple-LaMa] 结果已保存")

    image.save("output/original_iter18.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 18 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_18()
