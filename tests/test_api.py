"""
Tests for API endpoints
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def create_test_image_bytes(width=100, height=100):
    """Create bytes representation of a test image"""
    # Create a test image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return img_bytes.getvalue()


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    # Root returns HTML, not JSON
    assert "Watermark Remover" in response.text or "watermark" in response.text.lower()


def test_detect_watermark_endpoint():
    """Test watermark detection endpoint"""
    # Create test image bytes
    img_bytes = create_test_image_bytes()

    # Send request to /api/v1/process (process endpoint returns JSON response)
    response = client.post(
        "/api/v1/process",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"detection_method": "auto"},
    )

    # Should return 200 OK with JSON response
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") == True
    assert "output_url" in data or "download_url" in data


def test_remove_watermark_endpoint():
    """Test watermark removal endpoint"""
    # Create test image bytes
    img_bytes = create_test_image_bytes()

    # Send request to /api/v1/process
    response = client.post(
        "/api/v1/process",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"detection_method": "auto", "device": "cpu"},
    )

    # Should return 200 OK
    assert response.status_code == 200


def test_process_image_endpoint():
    """Test complete image processing endpoint"""
    # Create test image bytes
    img_bytes = create_test_image_bytes()

    # Send request to /api/v1/process
    response = client.post(
        "/api/v1/process",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"detection_method": "auto", "device": "cpu"},
    )

    # Should return 200 OK
    assert response.status_code == 200


def test_api_validation():
    """Test API parameter validation"""
    # Test with invalid device parameter
    img_bytes = create_test_image_bytes()

    response = client.post(
        "/api/v1/process",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"detection_method": "invalid_method", "device": "cpu"},
    )

    # Even with invalid method, it should still process (falling back to default)
    assert response.status_code == 200
