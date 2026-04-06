"""
PyTorch-based AI Inpainting Module
Uses a simple UNet-like architecture for image inpainting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Optional


class SimpleInpaintNet(nn.Module):
    """Simple UNet-like network for image inpainting"""

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(4, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, 3, padding=1)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 1024, 3, padding=1)

        # Decoder
        self.dec4 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dec3 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)

        # Output
        self.output = nn.Conv2d(64, 3, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # x shape: [B, 4, H, W] where 4 = RGB + mask
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        e3 = F.relu(self.enc3(self.pool(e2)))
        e4 = F.relu(self.enc4(self.pool(e3)))

        b = F.relu(self.bottleneck(self.pool(e4)))

        d4 = F.relu(self.dec4(self.upsample(b)))
        d4 = torch.cat([d4, e4], dim=1)

        d3 = F.relu(self.dec3(self.upsample(d4)))
        d3 = torch.cat([d3, e3], dim=1)

        d2 = F.relu(self.dec2(self.upsample(d3)))
        d2 = torch.cat([d2, e2], dim=1)

        d1 = F.relu(self.dec1(self.upsample(d2)))
        d1 = torch.cat([d1, e1], dim=1)

        out = torch.sigmoid(self.output(d1))
        return out


class PyTorchInpainter:
    """PyTorch-based AI inpainter"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = SimpleInpaintNet().to(device)
        self.model.eval()
        print(f"PyTorch inpainter initialized on {device}")

    def preprocess(self, image: Image.Image, mask: Image.Image) -> tuple:
        """Preprocess image and mask for model input"""
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

        # Create masked image
        masked_img = img_array * (1 - mask_array[:, :, None])

        # Concatenate image + mask
        input_tensor = np.concatenate([img_array, mask_array[:, :, None]], axis=-1)
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)

        return input_tensor.to(self.device), img_array, mask_array

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Perform AI inpainting"""
        with torch.no_grad():
            input_tensor, img_array, mask_array = self.preprocess(image, mask)

            # Run inference
            output = self.model(input_tensor)

            # Convert to numpy
            result = output.squeeze().permute(1, 2, 0).cpu().numpy()

            # Blend with original using mask
            mask_3ch = np.stack([mask_array] * 3, axis=-1)
            blended = result * mask_3ch + img_array * (1 - mask_3ch)

            # Convert to uint8
            result_uint8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

            return Image.fromarray(result_uint8)


def remove_watermark_ai(
    image: Image.Image,
    mask: Image.Image,
    device: str = "cpu",
    use_ai: bool = False,
) -> Image.Image:
    """
    Remove watermark using AI inpainting if available, fallback to OpenCV

    Args:
        image: Input PIL Image with watermarks
        mask: Mask indicating watermark areas
        device: Device to run on ("cpu" or "cuda")
        use_ai: Whether to use AI model (requires trained weights)

    Returns:
        PIL Image with watermarks removed
    """
    if use_ai:
        try:
            inpainter = PyTorchInpainter(device=device)
            return inpainter.inpaint(image, mask)
        except Exception as e:
            print(f"AI inpainting failed: {e}, falling back to OpenCV")

    # Fallback to OpenCV
    from inpainter.enhanced import remove_watermark
    return remove_watermark(image, mask, method="ns")
