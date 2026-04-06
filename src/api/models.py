"""
Pydantic models for API request/response validation
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionMethod(str, Enum):
    """Watermark detection methods"""

    AUTO = "auto"
    COLOR = "color"
    EDGE = "edge"
    CORNERS = "corners"
    PATTERN = "pattern"
    TEMPLATE = "template"
    TEXT = "text"
    CORNER_FOCUS = "corner_focus"


class InpaintMethod(str, Enum):
    """Inpainting methods"""

    TELEA = "telea"
    NS = "ns"
    NS_ORIGINAL = "ns_original"


class Device(str, Enum):
    """Processing device options"""

    CPU = "cpu"
    CUDA = "cuda"


# ============== Request Models ==============


class ProcessImageRequest(BaseModel):
    """Request model for image processing"""

    detection_method: DetectionMethod = Field(
        default=DetectionMethod.AUTO, description="Watermark detection method"
    )
    inpaint_method: InpaintMethod = Field(
        default=InpaintMethod.TELEA, description="Inpainting algorithm"
    )
    device: Device = Field(default=Device.CPU, description="Processing device")
    return_mask: bool = Field(
        default=False, description="Whether to return the detection mask"
    )


class ProcessVideoRequest(BaseModel):
    """Request model for video processing"""

    detection_method: DetectionMethod = Field(
        default=DetectionMethod.AUTO, description="Watermark detection method"
    )
    device: Device = Field(default=Device.CPU, description="Processing device")
    frame_range: Optional[List[int]] = Field(
        default=None, description="Frame range to process [start, end]"
    )
    quality: str = Field(default="high", description="Output quality: low, medium, high")


class BatchProcessRequest(BaseModel):
    """Request model for batch processing"""

    file_ids: List[str] = Field(..., description="List of uploaded file IDs")
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.AUTO, description="Watermark detection method"
    )
    inpaint_method: InpaintMethod = Field(
        default=InpaintMethod.TELEA, description="Inpainting algorithm"
    )
    device: Device = Field(default=Device.CPU, description="Processing device")


# ============== Response Models ==============


class ProcessingStatus(str, Enum):
    """Processing status options"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageInfo(BaseModel):
    """Image information"""

    width: int
    height: int
    format: str
    size_bytes: int
    mode: str


class VideoInfo(BaseModel):
    """Video information"""

    duration: float
    width: int
    height: int
    fps: float
    format: str
    size_bytes: int
    codec: str


class ProcessResult(BaseModel):
    """Processing result"""

    success: bool
    file_id: str
    output_url: Optional[str] = None
    mask_url: Optional[str] = None
    processing_time: float
    message: str = ""


class BatchResult(BaseModel):
    """Batch processing result"""

    batch_id: str
    total: int
    processed: int
    succeeded: int
    failed: int
    results: List[ProcessResult]
    status: ProcessingStatus


class JobStatus(BaseModel):
    """Job status response"""

    job_id: str
    status: ProcessingStatus
    progress: float  # 0-100
    created_at: datetime
    updated_at: datetime
    result: Optional[ProcessResult] = None
    error_message: Optional[str] = None


class APIResponse(BaseModel):
    """Standard API response wrapper"""

    success: bool
    message: str
    data: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    timestamp: datetime
    services: dict


class UsageStats(BaseModel):
    """Usage statistics"""

    total_processed: int
    today_processed: int
    total_users: int
    active_jobs: int


class PricingTier(BaseModel):
    """Pricing tier information"""

    name: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: dict


# ============== Upload Models ==============


class UploadResponse(BaseModel):
    """File upload response"""

    success: bool
    file_id: str
    filename: str
    size: int
    content_type: str
    upload_time: datetime


class FileInfo(BaseModel):
    """File information"""

    file_id: str
    filename: str
    content_type: str
    size: int
    upload_time: datetime
    status: str
    metadata: Optional[dict] = None
