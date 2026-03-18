"""Image Preprocessing & Filtering Pipeline.

Implements various image filtering techniques used to prepare lecture
slide captures and professor video frames for downstream ML training.

Techniques:
    1. Spatial Filtering   — Gaussian, Median, Bilateral, Box blur
    2. Edge Detection      — Canny, Sobel, Laplacian, Scharr
    3. Morphological Ops   — Erosion, Dilation, Opening, Closing, Gradient
    4. Frequency Filtering — Fourier Transform, High/Low pass
    5. Adaptive Processing — CLAHE, histogram equalization, gamma correction
    6. Noise Reduction     — Non-local means, Wiener filter approximation
    7. Perspective Correction — For projected slides (4-point transform)
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum


# ── Configuration ────────────────────────────────────────────────

class FilterType(Enum):
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    BOX = "box"
    CANNY = "canny"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    SCHARR = "scharr"
    CLAHE = "clahe"
    NLM_DENOISE = "nlm_denoise"
    MORPHOLOGICAL = "morphological"
    SHARPEN = "sharpen"


@dataclass
class PreprocessConfig:
    """Configuration for the image preprocessing pipeline."""
    target_size: Tuple[int, int] = (640, 480)
    # Gaussian blur
    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.0
    # Bilateral filter
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    # Canny edge detection
    canny_low: int = 50
    canny_high: int = 150
    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    # Morphological
    morph_kernel_size: int = 5
    # Non-local means denoising
    nlm_h: float = 10.0
    nlm_template_window: int = 7
    nlm_search_window: int = 21
    # Gamma correction
    gamma: float = 1.0
    # Pipeline steps (ordered list of filters to apply)
    pipeline: List[FilterType] = field(default_factory=lambda: [
        FilterType.CLAHE,
        FilterType.BILATERAL,
        FilterType.SHARPEN,
    ])


# ── Core Filtering Functions ─────────────────────────────────────

def apply_gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to smooth the image while preserving edges better than box blur.
    Uses a Gaussian kernel where center pixels have more weight than edge pixels."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply median filter — excellent for salt-and-pepper noise removal
    while preserving edges. Each pixel replaced by median of neighborhood."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(image: np.ndarray, d: int = 9,
                           sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter — smooths while keeping edges sharp.
    Uses both spatial distance and pixel intensity difference for weighting.
    Ideal for slide text preservation during noise removal."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_box_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Simple averaging filter. Each pixel = mean of its neighborhood.
    Fast but blurs edges. Used as baseline comparison."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.blur(image, (ksize, ksize))


def apply_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp masking — sharpens edges by subtracting blurred version.
    Enhances text readability in slide captures."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# ── Edge Detection ───────────────────────────────────────────────

def apply_canny_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection — multi-stage algorithm:
    1. Gaussian smoothing
    2. Gradient computation (Sobel)
    3. Non-maximum suppression
    4. Double thresholding + edge tracking by hysteresis
    
    Used to detect slide boundaries, text regions, and diagram outlines."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low, high)


def apply_sobel_edges(image: np.ndarray, ksize: int = 3,
                       direction: str = "both") -> np.ndarray:
    """Sobel operator — computes image gradient in x and/or y direction.
    Detects horizontal and vertical edges separately.
    Used for text line detection in slides."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if direction == "x":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    elif direction == "y":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    else:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return cv2.convertScaleAbs(magnitude)


def apply_laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Laplacian operator — second derivative edge detector.
    Detects edges in all directions simultaneously.
    Sensitive to noise, so often used after Gaussian smoothing."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)


def apply_scharr_edges(image: np.ndarray, direction: str = "both") -> np.ndarray:
    """Scharr operator — more accurate than 3x3 Sobel for gradient estimation.
    Better rotational symmetry in gradient computation."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if direction == "x":
        return cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_64F, 1, 0))
    elif direction == "y":
        return cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_64F, 0, 1))
    else:
        sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(np.sqrt(sx ** 2 + sy ** 2))


# ── Morphological Operations ────────────────────────────────────

def apply_morphological(image: np.ndarray, operation: str = "close",
                        ksize: int = 5, iterations: int = 1) -> np.ndarray:
    """Morphological operations for binary/grayscale images.
    
    Operations:
        erode   — shrinks bright regions (removes small bright noise)
        dilate  — expands bright regions (fills small dark gaps)
        open    — erode→dilate (removes small bright spots, keeps structure)
        close   — dilate→erode (fills small dark holes in bright regions)
        gradient — dilate−erode (extracts edges/boundaries)
        tophat  — original−opening (extracts bright details on dark bg)
        blackhat — closing−original (extracts dark details on bright bg)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    ops = {
        "erode": lambda: cv2.erode(image, kernel, iterations=iterations),
        "dilate": lambda: cv2.dilate(image, kernel, iterations=iterations),
        "open": lambda: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations),
        "close": lambda: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations),
        "gradient": lambda: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations),
        "tophat": lambda: cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations),
        "blackhat": lambda: cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations),
    }
    return ops.get(operation, ops["close"])()


# ── Contrast & Histogram Enhancement ────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast without over-amplifying noise.
    Critical for slide captures under variable lecture hall lighting."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization — spreads intensity values across
    full range. Improves contrast but can amplify noise."""
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction — adjusts image brightness nonlinearly.
    gamma < 1: brightens dark images (lecture halls)
    gamma > 1: darkens overexposed images (projector glare)"""
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


# ── Noise Reduction ──────────────────────────────────────────────

def apply_nlm_denoise(image: np.ndarray, h: float = 10.0,
                       template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Non-Local Means denoising — preserves fine details (text, equations)
    while removing Gaussian noise. Compares patches across image for denoising.
    Slower but much better quality than simple blur for OCR preprocessing."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                template_window, search_window)
    else:
        return cv2.fastNlMeansDenoising(image, None, h,
                                        template_window, search_window)


def apply_wiener_approx(image: np.ndarray, noise_var: float = 25.0) -> np.ndarray:
    """Approximate Wiener filter in frequency domain.
    Optimal for removing Gaussian noise while minimizing MSE.
    Useful for motion-blurred captures when professor moves camera."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float64)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2
    # Wiener filter: H* / (|H|^2 + NSR)  — assuming H=1 (identity degradation)
    wiener = power / (power + noise_var)
    filtered = fshift * wiener
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Frequency Domain Filtering ──────────────────────────────────

def apply_fourier_filter(image: np.ndarray, filter_type: str = "lowpass",
                         cutoff: int = 30) -> np.ndarray:
    """Frequency domain filtering using FFT.
    lowpass  — removes high-frequency noise, blurs
    highpass — extracts edges and fine details
    bandpass — isolates specific frequency range (text characters)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)

    # Create frequency mask
    mask = np.zeros((rows, cols), np.float64)
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    if filter_type == "lowpass":
        mask[dist <= cutoff] = 1.0
        # Smooth transition
        transition = (dist > cutoff) & (dist < cutoff * 1.5)
        mask[transition] = 1.0 - (dist[transition] - cutoff) / (cutoff * 0.5)
    elif filter_type == "highpass":
        mask[dist > cutoff] = 1.0
        transition = (dist <= cutoff) & (dist > cutoff * 0.5)
        mask[transition] = (dist[transition] - cutoff * 0.5) / (cutoff * 0.5)
    elif filter_type == "bandpass":
        low = cutoff
        high = cutoff * 3
        mask[(dist >= low) & (dist <= high)] = 1.0

    filtered = fshift * mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Perspective Correction ───────────────────────────────────────

def correct_perspective(image: np.ndarray,
                        src_points: Optional[np.ndarray] = None) -> np.ndarray:
    """Correct perspective distortion from projected slides.
    If src_points not given, auto-detect the largest rectangular contour.
    
    This transforms a trapezoidal slide capture → perfect rectangle,
    critical for accurate OCR on projected slides."""
    if src_points is not None:
        # Manual 4-point transform
        return _four_point_transform(image, src_points)

    # Auto-detect slide region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)

    # Dilate to connect edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return _four_point_transform(image, approx.reshape(4, 2))

    # No quadrilateral found — return original
    return image


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to warp 4 points into a rectangle."""
    rect = _order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect


# ── Slide Change Detection ───────────────────────────────────────

def detect_slide_change(frame_a: np.ndarray, frame_b: np.ndarray,
                        threshold: float = 0.35) -> Tuple[bool, float]:
    """Detect slide transitions using histogram comparison + frame differencing.
    
    Combines:
    1. Histogram correlation — measures global color distribution change
    2. Structural similarity — measures local structural change
    
    Returns: (is_change, similarity_score)"""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY) if len(frame_a.shape) == 3 else frame_a
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY) if len(frame_b.shape) == 3 else frame_b

    # Resize to same dimensions
    h, w = 240, 320
    gray_a = cv2.resize(gray_a, (w, h))
    gray_b = cv2.resize(gray_b, (w, h))

    # Histogram comparison
    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    # Frame differencing
    diff = cv2.absdiff(gray_a, gray_b)
    diff_score = np.mean(diff) / 255.0

    # Combined score (lower = more similar)
    similarity = hist_corr * (1.0 - diff_score)
    is_change = similarity < (1.0 - threshold)

    return is_change, float(similarity)


# ── Pipeline Executor ────────────────────────────────────────────

class ImagePreprocessor:
    """Configurable image preprocessing pipeline.
    Chains multiple filtering operations in sequence."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._filter_map = {
            FilterType.GAUSSIAN: lambda img: apply_gaussian_blur(
                img, self.config.gaussian_ksize, self.config.gaussian_sigma),
            FilterType.MEDIAN: lambda img: apply_median_filter(img),
            FilterType.BILATERAL: lambda img: apply_bilateral_filter(
                img, self.config.bilateral_d,
                self.config.bilateral_sigma_color, self.config.bilateral_sigma_space),
            FilterType.BOX: lambda img: apply_box_blur(img),
            FilterType.CANNY: lambda img: apply_canny_edges(
                img, self.config.canny_low, self.config.canny_high),
            FilterType.SOBEL: lambda img: apply_sobel_edges(img),
            FilterType.LAPLACIAN: lambda img: apply_laplacian_edges(img),
            FilterType.SCHARR: lambda img: apply_scharr_edges(img),
            FilterType.CLAHE: lambda img: apply_clahe(
                img, self.config.clahe_clip_limit, self.config.clahe_grid_size),
            FilterType.NLM_DENOISE: lambda img: apply_nlm_denoise(
                img, self.config.nlm_h,
                self.config.nlm_template_window, self.config.nlm_search_window),
            FilterType.MORPHOLOGICAL: lambda img: apply_morphological(
                img, "close", self.config.morph_kernel_size),
            FilterType.SHARPEN: lambda img: apply_sharpening(img),
        }

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on an image."""
        result = image.copy()

        # Resize to target
        if self.config.target_size:
            result = cv2.resize(result, self.config.target_size)

        # Gamma correction
        if self.config.gamma != 1.0:
            result = apply_gamma_correction(result, self.config.gamma)

        # Apply each filter in the pipeline
        for filter_type in self.config.pipeline:
            fn = self._filter_map.get(filter_type)
            if fn:
                result = fn(result)

        return result

    def process_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Optimized preprocessing specifically for OCR text extraction.
        Pipeline: resize → CLAHE → bilateral → sharpen → binarize."""
        img = cv2.resize(image, self.config.target_size) if self.config.target_size else image.copy()
        img = apply_clahe(img, clip_limit=3.0)
        img = apply_bilateral_filter(img, d=9, sigma_color=50, sigma_space=50)
        img = apply_sharpening(img, strength=1.5)
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def process_for_gesture(self, image: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for gesture/pose detection.
        Pipeline: resize → denoise → CLAHE → normalize."""
        img = cv2.resize(image, (256, 256))
        img = apply_bilateral_filter(img, d=5, sigma_color=40, sigma_space=40)
        img = apply_clahe(img, clip_limit=2.0)
        # Normalize to [0, 1] range for neural network input
        return img.astype(np.float32) / 255.0


# ── Convenience ──────────────────────────────────────────────────

def preprocess_slide_capture(image: np.ndarray) -> np.ndarray:
    """Full slide preprocessing: perspective correct → enhance → OCR-ready."""
    corrected = correct_perspective(image)
    preprocessor = ImagePreprocessor(PreprocessConfig(
        target_size=(1280, 960),
        pipeline=[FilterType.CLAHE, FilterType.BILATERAL, FilterType.SHARPEN],
    ))
    return preprocessor.process(corrected)


def preprocess_professor_frame(image: np.ndarray) -> np.ndarray:
    """Preprocess a professor video frame for gesture detection."""
    preprocessor = ImagePreprocessor()
    return preprocessor.process_for_gesture(image)
