// Watermark Remover - Frontend Application
// Follows UI/UX best practices: touch targets, accessibility, reduced motion

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');
    const processingStatus = document.getElementById('processingStatus');
    const demoActions = document.getElementById('demoActions');
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    let processedBlob = null;

    // Mobile Navigation Toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', !isExpanded);
            navLinks.classList.toggle('active');
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
            });
        });

        // Close menu on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navLinks.classList.contains('active')) {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
                navToggle.focus();
            }
        });
    }

    // Upload area click handler
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Upload area keyboard handler
    uploadArea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--color-primary)';
        uploadArea.style.background = 'var(--primary-50)';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--color-border)';
        uploadArea.style.background = 'var(--color-bg-secondary)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--color-border)';
        uploadArea.style.background = 'var(--color-bg-secondary)';

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle file upload
    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('请上传图片文件');
            return;
        }

        // Show preview with original image
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
            previewContainer.style.display = 'grid';
            uploadArea.style.display = 'none';

            // Process the image
            processImage(file);
        };
        reader.readAsDataURL(file);
    }

    // Process image through API
    async function processImage(file) {
        // Show processing status
        processingStatus.style.display = 'block';
        previewContainer.style.display = 'none';
        demoActions.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('detection_method', 'auto');
            formData.append('device', 'cpu');

            const response = await fetch('/process/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || 'Processing failed');
            }

            // Get the processed image blob
            processedBlob = await response.blob();
            const url = URL.createObjectURL(processedBlob);

            processedImage.src = url;

            // Show results
            processingStatus.style.display = 'none';
            previewContainer.style.display = 'grid';
            demoActions.style.display = 'flex';

        } catch (error) {
            console.error('Error processing image:', error);
            processingStatus.style.display = 'none';
            previewContainer.style.display = 'none';
            uploadArea.style.display = 'block';
            alert('处理失败：' + error.message + '，请重试');
        }
    }

    // Download handler
    downloadBtn.addEventListener('click', () => {
        if (processedBlob) {
            const url = URL.createObjectURL(processedBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'watermark_removed_' + Date.now() + '.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    });

    // Reset handler
    resetBtn.addEventListener('click', () => {
        fileInput.value = '';
        originalImage.src = '';
        processedImage.src = '';
        processedBlob = null;
        previewContainer.style.display = 'none';
        demoActions.style.display = 'none';
        uploadArea.style.display = 'block';
        fileInput.focus();
    });

    // Smooth scroll for navigation links with offset for fixed navbar
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: document.getElementById('main-content')?.matches(':focus-within') ||
                        window.matchMedia('(prefers-reduced-motion: reduce)').matches
                        ? 'auto'
                        : 'smooth'
                });
            }
        });
    });

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    let lastScrollY = window.scrollY;
    let ticking = false;

    function updateNavbar() {
        if (window.scrollY > 10) {
            navbar.style.boxShadow = 'var(--shadow)';
        } else {
            navbar.style.boxShadow = 'none';
        }
        lastScrollY = window.scrollY;
        ticking = false;
    }

    window.addEventListener('scroll', () => {
        if (!ticking) {
            window.requestAnimationFrame(updateNavbar);
            ticking = true;
        }
    }, { passive: true });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                fadeObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe feature cards and steps
    document.querySelectorAll('.feature-card, .step').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        fadeObserver.observe(el);
    });
});
