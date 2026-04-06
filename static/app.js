// Watermark Remover - Premium Frontend Application
// Modern, Professional, Accessible

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
    const themeToggle = document.querySelector('.theme-toggle');
    const navbar = document.querySelector('.navbar');
    const faqItems = document.querySelectorAll('.faq-item');

    let processedBlob = null;
    let originalFileName = '';

    // ===================================
    // Theme Toggle
    // ===================================
    const THEME_KEY = 'watermark-remover-theme';

    function getPreferredTheme() {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved) return saved;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (metaThemeColor) {
            metaThemeColor.setAttribute('content', theme === 'dark' ? '#020617' : '#3b82f6');
        }
    }

    applyTheme(getPreferredTheme());

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem(THEME_KEY, newTheme);
        });
    }

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem(THEME_KEY)) {
            applyTheme(e.matches ? 'dark' : 'light');
        }
    });

    // ===================================
    // Mobile Navigation
    // ===================================
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', !isExpanded);
            navLinks.classList.toggle('active');
        });

        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navLinks.classList.contains('active')) {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
                navToggle.focus();
            }
        });
    }

    // ===================================
    // Navbar Scroll Effect
    // ===================================
    let lastScrollY = window.scrollY;
    let ticking = false;

    function updateNavbar() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
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

    // ===================================
    // FAQ Accordion
    // ===================================
    faqItems.forEach((item, index) => {
        const question = item.querySelector('.faq-question');
        question.addEventListener('click', () => {
            const isExpanded = item.getAttribute('aria-expanded') === 'true';

            // Close all other items
            faqItems.forEach((otherItem, otherIndex) => {
                if (otherIndex !== index) {
                    otherItem.setAttribute('aria-expanded', 'false');
                }
            });

            // Toggle current item
            item.setAttribute('aria-expanded', !isExpanded);
        });
    });

    // ===================================
    // Scroll Animations
    // ===================================
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

    // Observe elements
    document.querySelectorAll('.feature-card, .step-card, .pricing-card, .faq-item').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        fadeObserver.observe(el);
    });

    // ===================================
    // Smooth Scroll
    // ===================================
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
                    behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
                });
            }
        });
    });

    // ===================================
    // File Upload & Processing
    // ===================================
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-primary)';
            uploadArea.style.background = 'var(--primary-50)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-border)';
            uploadArea.style.background = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-border)';
            uploadArea.style.background = '';

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                handleFile(files[0]);
            }
        });
    }

    fileInput?.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('请上传图片文件');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            alert('文件大小超过 10MB 限制');
            return;
        }

        originalFileName = file.name.replace(/\.[^/.]+$/, '');
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
            document.getElementById('originalSize').textContent = formatFileSize(file.size);
            previewContainer.style.display = 'grid';
            uploadArea.style.display = 'none';
            processImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function processImage(file) {
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
                throw new Error(errorData.message || '处理失败');
            }

            processedBlob = await response.blob();
            const url = URL.createObjectURL(processedBlob);
            processedImage.src = url;

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

    downloadBtn?.addEventListener('click', () => {
        if (processedBlob) {
            const url = URL.createObjectURL(processedBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${originalFileName}_no_watermark.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    });

    resetBtn?.addEventListener('click', () => {
        fileInput.value = '';
        originalImage.src = '';
        processedImage.src = '';
        processedBlob = null;
        previewContainer.style.display = 'none';
        demoActions.style.display = 'none';
        uploadArea.style.display = 'block';
        fileInput.focus();
    });

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
});
