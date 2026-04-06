/**
 * Video Watermark Remover - Frontend Logic
 * Handles video upload, processing, and comparison viewer
 */

// State management
const state = {
    file: null,
    fileId: null,
    jobId: null,
    isProcessing: false,
    theme: 'light',
};

// DOM Elements
const elements = {
    // Upload
    uploadZone: null,
    fileInput: null,
    uploadState: null,
    settingsPanel: null,
    processingState: null,
    resultState: null,

    // Video Info
    videoPreview: null,
    playPreviewBtn: null,
    videoName: null,
    videoSize: null,
    videoDuration: null,
    videoResolution: null,

    // Settings
    detectionMethod: null,
    frameInterval: null,
    intervalValue: null,
    startProcessBtn: null,

    // Processing
    progressFill: null,
    progressPercent: null,
    progressEta: null,
    originalFrame: null,
    processedFrame: null,

    // Result
    originalVideo: null,
    processedVideo: null,
    comparisonSlider: null,
    sliderHandle: null,
    downloadBtn: null,
    replayBtn: null,

    // Theme
    themeToggle: null,
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeTheme();
    initializeUploadHandlers();
    initializeSettingsHandlers();
    initializeComparisonSlider();
});

/**
 * Initialize all DOM element references
 */
function initializeElements() {
    // Upload elements
    elements.uploadZone = document.getElementById('uploadZone');
    elements.fileInput = document.getElementById('fileInput');
    elements.uploadState = document.getElementById('uploadState');
    elements.settingsPanel = document.getElementById('settingsPanel');
    elements.processingState = document.getElementById('processingState');
    elements.resultState = document.getElementById('resultState');

    // Video info
    elements.videoPreview = document.getElementById('videoPreview');
    elements.playPreviewBtn = document.getElementById('playPreviewBtn');
    elements.videoName = document.getElementById('videoName');
    elements.videoSize = document.getElementById('videoSize');
    elements.videoDuration = document.getElementById('videoDuration');
    elements.videoResolution = document.getElementById('videoResolution');

    // Settings
    elements.detectionMethod = document.getElementById('detectionMethod');
    elements.frameInterval = document.getElementById('frameInterval');
    elements.intervalValue = document.getElementById('intervalValue');
    elements.startProcessBtn = document.getElementById('startProcessBtn');

    // Processing
    elements.progressFill = document.getElementById('progressFill');
    elements.progressPercent = document.getElementById('progressPercent');
    elements.progressEta = document.getElementById('progressEta');
    elements.originalFrame = document.getElementById('originalFrame');
    elements.processedFrame = document.getElementById('processedFrame');

    // Result
    elements.originalVideo = document.getElementById('originalVideo');
    elements.processedVideo = document.getElementById('processedVideo');
    elements.comparisonSlider = document.getElementById('comparisonSlider');
    elements.sliderHandle = document.getElementById('sliderHandle');
    elements.downloadBtn = document.getElementById('downloadBtn');
    elements.replayBtn = document.getElementById('replayBtn');

    // Theme
    elements.themeToggle = document.getElementById('themeToggle');
}

/**
 * Initialize theme toggle functionality
 */
function initializeTheme() {
    // Check for saved theme preference or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);

    elements.themeToggle?.addEventListener('click', () => {
        const newTheme = state.theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    });
}

function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

/**
 * Initialize upload zone handlers
 */
function initializeUploadHandlers() {
    // Click to upload
    elements.uploadZone?.addEventListener('click', () => {
        elements.fileInput?.click();
    });

    // File input change
    elements.fileInput?.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // Drag and drop
    elements.uploadZone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone?.classList.add('drag-over');
    });

    elements.uploadZone?.addEventListener('dragleave', () => {
        elements.uploadZone?.classList.remove('drag-over');
    });

    elements.uploadZone?.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadZone?.classList.remove('drag-over');
        const file = e.dataTransfer?.files?.[0];
        if (file && file.type.startsWith('video/')) {
            handleFileSelect(file);
        }
    });

    // Video preview controls
    elements.videoPreview?.addEventListener('loadedmetadata', () => {
        updateVideoInfo();
    });

    elements.playPreviewBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleVideoPreview();
    });
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showError('请选择有效的视频文件');
        return;
    }

    // Validate file size (100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('文件大小超过 100MB 限制');
        return;
    }

    state.file = file;

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    if (elements.videoPreview) {
        elements.videoPreview.src = previewUrl;
    }

    // Update file name
    if (elements.videoName) {
        elements.videoName.textContent = file.name;
    }

    // Show settings panel
    showSettingsPanel();
}

/**
 * Update video metadata display
 */
function updateVideoInfo() {
    const video = elements.videoPreview;
    if (!video) return;

    // Size
    if (elements.videoSize && state.file) {
        elements.videoSize.textContent = formatFileSize(state.file.size);
    }

    // Duration
    if (elements.videoDuration && video.duration) {
        elements.videoDuration.textContent = formatDuration(video.duration);
    }

    // Resolution
    if (elements.videoResolution && video.videoWidth && video.videoHeight) {
        elements.videoResolution.textContent = `${video.videoWidth}×${video.videoHeight}`;
    }
}

/**
 * Toggle video preview playback
 */
function toggleVideoPreview() {
    const video = elements.videoPreview;
    const btn = elements.playPreviewBtn;
    if (!video || !btn) return;

    if (video.paused) {
        video.play();
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
            </svg>
        `;
    } else {
        video.pause();
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
        `;
    }
}

/**
 * Show settings panel after file upload
 */
function showSettingsPanel() {
    elements.uploadState?.classList.add('hidden');
    elements.settingsPanel?.classList.remove('hidden');
    elements.processingState?.classList.add('hidden');
    elements.resultState?.classList.add('hidden');
}

/**
 * Initialize settings handlers
 */
function initializeSettingsHandlers() {
    // Frame interval slider
    elements.frameInterval?.addEventListener('input', (e) => {
        const value = e.target.value;
        if (elements.intervalValue) {
            elements.intervalValue.textContent = `每${value}帧`;
        }
    });

    // Start processing button
    elements.startProcessBtn?.addEventListener('click', () => {
        startProcessing();
    });
}

/**
 * Start video processing
 */
async function startProcessing() {
    if (!state.file) return;

    state.isProcessing = true;

    // Show processing state
    elements.settingsPanel?.classList.add('hidden');
    elements.processingState?.classList.remove('hidden');

    // Get settings
    const detectionMethod = elements.detectionMethod?.value || 'auto';
    const quality = document.querySelector('input[name="quality"]:checked')?.value || 'medium';
    const frameInterval = elements.frameInterval?.value || 1;

    try {
        // Upload video
        const uploadResult = await uploadVideo(state.file);
        if (!uploadResult.success) {
            throw new Error(uploadResult.message || '上传失败');
        }

        state.fileId = uploadResult.file_id;

        // Start processing
        const processResult = await startVideoProcess(state.fileId, detectionMethod, frameInterval, quality);
        if (!processResult.success) {
            throw new Error(processResult.message || '处理失败');
        }

        state.jobId = processResult.job_id;

        // Poll for progress
        pollProgress(state.jobId);

    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message || '处理过程中发生错误');
        state.isProcessing = false;
    }
}

/**
 * Upload video file
 */
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/v1/upload-video', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        return { success: false, message: error.detail || '上传失败' };
    }

    return await response.json();
}

/**
 * Start video processing
 */
async function startVideoProcess(fileId, detectionMethod, frameInterval, quality) {
    const formData = new FormData();
    formData.append('detection_method', detectionMethod);
    formData.append('frame_interval', frameInterval);
    formData.append('quality', quality);

    const response = await fetch(`/api/v1/process-video/${fileId}`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        return { success: false, message: error.detail || '处理失败' };
    }

    return await response.json();
}

/**
 * Poll processing progress
 */
async function pollProgress(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/v1/video-status/${jobId}`);
            if (!response.ok) {
                clearInterval(pollInterval);
                throw new Error('无法获取进度');
            }

            const data = await response.json();
            updateProgress(data.progress, data.status, data.output_url);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showResult(data.output_url);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                throw new Error(data.error || '处理失败');
            }
        } catch (error) {
            console.error('Poll error:', error);
            clearInterval(pollInterval);
            showError(error.message);
            state.isProcessing = false;
        }
    }, 1000);
}

/**
 * Update progress display
 */
function updateProgress(progress, status, outputUrl) {
    if (elements.progressFill) {
        elements.progressFill.style.width = `${Math.min(progress, 100)}%`;
    }

    if (elements.progressPercent) {
        elements.progressPercent.textContent = `${Math.round(progress)}%`;
    }

    if (elements.progressEta) {
        const remaining = (100 - progress) / 100 * 30; // Rough estimate
        if (remaining < 60) {
            elements.progressEta.textContent = `剩余时间：约${Math.round(remaining)}秒`;
        } else {
            const minutes = Math.round(remaining / 60);
            elements.progressEta.textContent = `剩余时间：约${minutes}分钟`;
        }
    }

    // Update frame previews (simulated)
    updateFramePreviews(progress);
}

/**
 * Update frame preview canvases (simulated)
 */
function updateFramePreviews(progress) {
    // This would show actual frames in a real implementation
    // For now, just show placeholder colors
    const originalCtx = elements.originalFrame?.getContext('2d');
    const processedCtx = elements.processedFrame?.getContext('2d');

    if (originalCtx && elements.originalFrame) {
        originalCtx.fillStyle = '#1e293b';
        originalCtx.fillRect(0, 0, elements.originalFrame.width, elements.originalFrame.height);
        originalCtx.fillStyle = '#64748b';
        originalCtx.font = '12px sans-serif';
        originalCtx.textAlign = 'center';
        originalCtx.fillText('原始帧', elements.originalFrame.width / 2, elements.originalFrame.height / 2);
    }

    if (processedCtx && elements.processedFrame) {
        const gradient = processedCtx.createLinearGradient(0, 0, elements.processedFrame.width, elements.processedFrame.height);
        gradient.addColorStop(0, '#3b82f6');
        gradient.addColorStop(1, '#8b5cf6');
        processedCtx.fillStyle = gradient;
        processedCtx.fillRect(0, 0, elements.processedFrame.width, elements.processedFrame.height);
        processedCtx.fillStyle = '#fff';
        processedCtx.font = '12px sans-serif';
        processedCtx.textAlign = 'center';
        processedCtx.fillText('处理中...', elements.processedFrame.width / 2, elements.processedFrame.height / 2);
    }
}

/**
 * Show result state
 */
function showResult(outputUrl) {
    elements.processingState?.classList.add('hidden');
    elements.resultState?.classList.remove('hidden');

    // Setup download button
    if (elements.downloadBtn && outputUrl) {
        elements.downloadBtn.href = outputUrl;
    }

    // Setup replay button
    elements.replayBtn?.addEventListener('click', () => {
        resetState();
    });

    // For demo purposes, load the same video as both original and processed
    // In a real implementation, you would load the actual processed video
    if (state.file && elements.originalVideo) {
        const url = URL.createObjectURL(state.file);
        elements.originalVideo.src = url;
        elements.processedVideo.src = url; // Placeholder
    }
}

/**
 * Initialize comparison slider
 */
function initializeComparisonSlider() {
    const slider = elements.comparisonSlider;
    const handle = elements.sliderHandle;
    const originalImage = slider?.querySelector('.comparison-image.original');

    if (!slider || !handle || !originalImage) return;

    let isDragging = false;

    const updateSlider = (clientX) => {
        const rect = slider.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const percentage = (x / rect.width) * 100;

        handle.style.left = `${percentage}%`;
        originalImage.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;
    };

    // Mouse events
    handle.addEventListener('mousedown', () => {
        isDragging = true;
        handle.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            e.preventDefault();
            updateSlider(e.clientX);
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        handle.style.cursor = 'grab';
    });

    // Touch events
    handle.addEventListener('touchstart', () => {
        isDragging = true;
    });

    slider.addEventListener('touchmove', (e) => {
        if (isDragging) {
            const touch = e.touches[0];
            updateSlider(touch.clientX);
        }
    });

    slider.addEventListener('touchend', () => {
        isDragging = false;
    });

    // Click to jump
    slider.addEventListener('click', (e) => {
        updateSlider(e.clientX);
    });
}

/**
 * Reset state for new upload
 */
function resetState() {
    state.file = null;
    state.fileId = null;
    state.jobId = null;
    state.isProcessing = false;

    // Reset UI
    elements.uploadState?.classList.remove('hidden');
    elements.settingsPanel?.classList.add('hidden');
    elements.processingState?.classList.add('hidden');
    elements.resultState?.classList.add('hidden');

    // Reset file input
    if (elements.fileInput) {
        elements.fileInput.value = '';
    }

    // Reset progress
    if (elements.progressFill) {
        elements.progressFill.style.width = '0%';
    }
    if (elements.progressPercent) {
        elements.progressPercent.textContent = '0%';
    }

    // Clear video sources
    if (elements.videoPreview) {
        elements.videoPreview.src = '';
    }
    if (elements.originalVideo) {
        elements.originalVideo.src = '';
    }
    if (elements.processedVideo) {
        elements.processedVideo.src = '';
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>${message}</span>
    `;

    // Add styles
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%) translateY(100px);
        background: #ef4444;
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 500;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        transition: transform 0.3s ease;
    `;

    toast.querySelector('svg')?.setAttribute('style', 'width: 24px; height: 24px;');

    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // Remove after 5 seconds
    setTimeout(() => {
        toast.style.transform = 'translateX(-50%) translateY(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration
 */
function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}
