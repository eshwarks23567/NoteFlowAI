"""Data Augmentation Pipeline for Training Data Generation.

Applies various transformations to increase training set diversity
and improve model robustness for lecture-specific conditions.

Categories:
    1. Geometric      — Rotation, flip, scale, translate, shear, elastic
    2. Photometric     — Brightness, contrast, saturation, hue, gamma
    3. Noise Injection — Gaussian, salt-and-pepper, speckle, Poisson
    4. Domain-Specific — Projector artifacts, lighting gradients, motion blur
    5. Occlusion       — Random erasing, cutout, professor body overlap
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import random


class AugmentationType(Enum):
    # Geometric
    ROTATE = "rotate"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    ELASTIC = "elastic"
    PERSPECTIVE = "perspective"
    # Photometric
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE_SHIFT = "hue_shift"
    GAMMA = "gamma"
    COLOR_JITTER = "color_jitter"
    # Noise
    GAUSSIAN_NOISE = "gaussian_noise"
    SALT_PEPPER = "salt_pepper"
    SPECKLE = "speckle"
    POISSON = "poisson"
    # Domain-specific
    MOTION_BLUR = "motion_blur"
    PROJECTOR_ARTIFACT = "projector_artifact"
    LIGHTING_GRADIENT = "lighting_gradient"
    JPEG_COMPRESSION = "jpeg_compression"
    # Occlusion
    RANDOM_ERASING = "random_erasing"
    CUTOUT = "cutout"


@dataclass
class AugmentConfig:
    """Configuration for augmentation pipeline."""
    # Probability of applying each augmentation
    probability: float = 0.5
    # Geometric params
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translate_range: Tuple[float, float] = (-0.1, 0.1)  # fraction of image size
    shear_range: Tuple[float, float] = (-10.0, 10.0)
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    # Photometric params
    brightness_range: Tuple[float, float] = (-30.0, 30.0)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift_range: Tuple[int, int] = (-10, 10)
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    # Noise params
    gaussian_noise_std: Tuple[float, float] = (5.0, 25.0)
    salt_pepper_ratio: float = 0.02
    # Domain-specific
    motion_blur_ksize: Tuple[int, int] = (5, 15)
    jpeg_quality_range: Tuple[int, int] = (30, 80)
    # Occlusion
    erasing_area_ratio: Tuple[float, float] = (0.02, 0.15)
    cutout_size: Tuple[int, int] = (30, 60)
    # Pipeline ordering
    augmentations: List[AugmentationType] = field(default_factory=lambda: [
        AugmentationType.ROTATE,
        AugmentationType.BRIGHTNESS,
        AugmentationType.CONTRAST,
        AugmentationType.GAUSSIAN_NOISE,
    ])


# ── Geometric Augmentations ─────────────────────────────────────

def augment_rotation(image: np.ndarray, angle: Optional[float] = None,
                     config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Rotate image by a random angle within range.
    Simulates camera tilt common in lecture hall setups."""
    cfg = config or AugmentConfig()
    if angle is None:
        angle = random.uniform(*cfg.rotation_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_flip(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Flip image horizontally or vertically.
    Horizontal flip doubles gesture training data."""
    if direction == "horizontal":
        return cv2.flip(image, 1)
    elif direction == "vertical":
        return cv2.flip(image, 0)
    else:
        return cv2.flip(image, -1)  # both


def augment_scale(image: np.ndarray, factor: Optional[float] = None,
                  config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random scale (zoom in/out). Simulates varying camera distances."""
    cfg = config or AugmentConfig()
    if factor is None:
        factor = random.uniform(*cfg.scale_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(image, (new_w, new_h))

    # Crop or pad to original size
    if factor > 1.0:
        # Crop center
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return scaled[start_y:start_y + h, start_x:start_x + w]
    else:
        # Pad with reflection
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        return cv2.copyMakeBorder(scaled, pad_y, h - new_h - pad_y,
                                   pad_x, w - new_w - pad_x,
                                   cv2.BORDER_REFLECT)


def augment_translate(image: np.ndarray, tx: Optional[float] = None,
                      ty: Optional[float] = None,
                      config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random translation. Simulates camera panning."""
    cfg = config or AugmentConfig()
    h, w = image.shape[:2]
    if tx is None:
        tx = random.uniform(*cfg.translate_range) * w
    if ty is None:
        ty = random.uniform(*cfg.translate_range) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_shear(image: np.ndarray, angle: Optional[float] = None,
                  config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random shear transformation. Simulates oblique camera angles."""
    cfg = config or AugmentConfig()
    if angle is None:
        angle = random.uniform(*cfg.shear_range)
    h, w = image.shape[:2]
    shear_rad = np.radians(angle)
    M = np.float32([[1, shear_rad, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_elastic(image: np.ndarray, alpha: float = 50.0,
                    sigma: float = 5.0) -> np.ndarray:
    """Elastic deformation — simulates lens distortion and handwriting variations.
    Applies random displacement field smoothed by Gaussian."""
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma
    ) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def augment_perspective(image: np.ndarray, magnitude: float = 0.05) -> np.ndarray:
    """Random perspective warp — simulates different viewpoints of projected slides."""
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offset = lambda: random.uniform(-magnitude * min(w, h), magnitude * min(w, h))
    pts2 = np.float32([
        [offset(), offset()],
        [w + offset(), offset()],
        [w + offset(), h + offset()],
        [offset(), h + offset()],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ── Photometric Augmentations ───────────────────────────────────

def augment_brightness(image: np.ndarray, delta: Optional[float] = None,
                       config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random brightness adjustment. Simulates variable lecture hall lighting."""
    cfg = config or AugmentConfig()
    if delta is None:
        delta = random.uniform(*cfg.brightness_range)
    result = image.astype(np.float32) + delta
    return np.clip(result, 0, 255).astype(np.uint8)


def augment_contrast(image: np.ndarray, factor: Optional[float] = None,
                     config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random contrast adjustment. Handles projector washout."""
    cfg = config or AugmentConfig()
    if factor is None:
        factor = random.uniform(*cfg.contrast_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    result = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def augment_saturation(image: np.ndarray, factor: Optional[float] = None,
                       config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random saturation adjustment."""
    cfg = config or AugmentConfig()
    if factor is None:
        factor = random.uniform(*cfg.saturation_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_hue_shift(image: np.ndarray, shift: Optional[int] = None,
                      config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random hue shift. Simulates color temperature variations."""
    cfg = config or AugmentConfig()
    if shift is None:
        shift = random.randint(*cfg.hue_shift_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_color_jitter(image: np.ndarray,
                         config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Combined color jitter — applies random brightness, contrast,
    saturation, and hue shift together."""
    result = augment_brightness(image, config=config)
    result = augment_contrast(result, config=config)
    result = augment_saturation(result, config=config)
    result = augment_hue_shift(result, config=config)
    return result


# ── Noise Injection ─────────────────────────────────────────────

def augment_gaussian_noise(image: np.ndarray, std: Optional[float] = None,
                           config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Add Gaussian noise — simulates camera sensor noise in low-light lecture halls."""
    cfg = config or AugmentConfig()
    if std is None:
        std = random.uniform(*cfg.gaussian_noise_std)
    noise = np.random.randn(*image.shape).astype(np.float32) * std
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_salt_pepper(image: np.ndarray, ratio: Optional[float] = None,
                        config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Add salt-and-pepper noise — simulates sensor defects and transmission artifacts."""
    cfg = config or AugmentConfig()
    if ratio is None:
        ratio = cfg.salt_pepper_ratio
    result = image.copy()
    num_salt = int(ratio * image.size * 0.5)
    num_pepper = int(ratio * image.size * 0.5)

    # Salt (white)
    coords = tuple(np.random.randint(0, d, num_salt) for d in image.shape[:2])
    result[coords[0], coords[1]] = 255

    # Pepper (black)
    coords = tuple(np.random.randint(0, d, num_pepper) for d in image.shape[:2])
    result[coords[0], coords[1]] = 0

    return result


def augment_speckle_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """Add multiplicative speckle noise — simulates projector artifacts."""
    noise = np.random.randn(*image.shape).astype(np.float32) * intensity
    noisy = image.astype(np.float32) * (1 + noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_poisson_noise(image: np.ndarray) -> np.ndarray:
    """Add Poisson noise — physically accurate model for photon counting noise."""
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(max(vals, 1)))
    noisy = np.random.poisson(image.astype(np.float64) * vals / 255.0) / vals * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ── Domain-Specific Augmentations ────────────────────────────────

def augment_motion_blur(image: np.ndarray, ksize: Optional[int] = None,
                        angle: Optional[float] = None,
                        config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Simulates motion blur from camera shake or professor movement."""
    cfg = config or AugmentConfig()
    if ksize is None:
        ksize = random.randint(*cfg.motion_blur_ksize)
    if angle is None:
        angle = random.uniform(0, 180)

    # Create motion blur kernel
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = 1.0 / ksize

    # Rotate kernel
    center = (ksize // 2, ksize // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def augment_projector_artifact(image: np.ndarray) -> np.ndarray:
    """Simulates common projector artifacts:
    - Hotspot (bright center, dim edges)
    - Color fringing
    - Keystoning brightness gradient"""
    h, w = image.shape[:2]
    result = image.astype(np.float32)

    # Vignette effect (bright center, dark edges)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - 0.3 * (dist / max_dist) ** 2

    if len(image.shape) == 3:
        for c in range(3):
            result[:, :, c] *= vignette
    else:
        result *= vignette

    # Random brightness gradient (simulating uneven projection)
    gradient = np.linspace(0.85, 1.15, w).reshape(1, -1)
    if random.random() > 0.5:
        gradient = np.linspace(0.85, 1.15, h).reshape(-1, 1)

    if len(image.shape) == 3:
        for c in range(3):
            result[:, :, c] *= gradient
    else:
        result *= gradient

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_lighting_gradient(image: np.ndarray) -> np.ndarray:
    """Simulates non-uniform lecture hall lighting — one side brighter than other."""
    h, w = image.shape[:2]
    direction = random.choice(["left_right", "top_bottom", "diagonal"])
    intensity = random.uniform(0.15, 0.35)

    if direction == "left_right":
        gradient = np.linspace(1 - intensity, 1 + intensity, w).reshape(1, -1)
    elif direction == "top_bottom":
        gradient = np.linspace(1 - intensity, 1 + intensity, h).reshape(-1, 1)
    else:
        gx = np.linspace(0, 1, w).reshape(1, -1)
        gy = np.linspace(0, 1, h).reshape(-1, 1)
        gradient = 1 - intensity + 2 * intensity * (gx + gy) / 2

    result = image.astype(np.float32)
    if len(image.shape) == 3:
        for c in range(3):
            result[:, :, c] *= gradient
    else:
        result *= gradient

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_jpeg_compression(image: np.ndarray, quality: Optional[int] = None,
                              config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Simulate JPEG compression artifacts (common in video streaming)."""
    cfg = config or AugmentConfig()
    if quality is None:
        quality = random.randint(*cfg.jpeg_quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


# ── Occlusion Augmentations ─────────────────────────────────────

def augment_random_erasing(image: np.ndarray,
                           config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Random erasing — occludes a random rectangular patch.
    Simulates professor standing in front of slides."""
    cfg = config or AugmentConfig()
    h, w = image.shape[:2]
    area = h * w
    target_ratio = random.uniform(*cfg.erasing_area_ratio)
    erase_area = int(area * target_ratio)

    aspect_ratio = random.uniform(0.3, 3.3)
    eh = int(np.sqrt(erase_area * aspect_ratio))
    ew = int(np.sqrt(erase_area / aspect_ratio))

    if eh >= h or ew >= w:
        return image

    x = random.randint(0, w - ew)
    y = random.randint(0, h - eh)

    result = image.copy()
    # Fill with random values or mean
    if random.random() > 0.5:
        result[y:y + eh, x:x + ew] = np.random.randint(0, 256, result[y:y + eh, x:x + ew].shape, dtype=np.uint8)
    else:
        result[y:y + eh, x:x + ew] = int(np.mean(image))

    return result


def augment_cutout(image: np.ndarray, num_holes: int = 1,
                   config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Cutout — occludes square patches with zeros. Improves model robustness
    to partial occlusion of slides and professor."""
    cfg = config or AugmentConfig()
    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(num_holes):
        size = random.randint(*cfg.cutout_size)
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, cx + size // 2)
        y2 = min(h, cy + size // 2)
        result[y1:y2, x1:x2] = 0

    return result


# ── Pipeline Executor ────────────────────────────────────────────

class DataAugmentor:
    """Configurable data augmentation pipeline."""

    def __init__(self, config: Optional[AugmentConfig] = None):
        self.config = config or AugmentConfig()
        self._augment_map = {
            AugmentationType.ROTATE: lambda img: augment_rotation(img, config=self.config),
            AugmentationType.FLIP_H: lambda img: augment_flip(img, "horizontal"),
            AugmentationType.FLIP_V: lambda img: augment_flip(img, "vertical"),
            AugmentationType.SCALE: lambda img: augment_scale(img, config=self.config),
            AugmentationType.TRANSLATE: lambda img: augment_translate(img, config=self.config),
            AugmentationType.SHEAR: lambda img: augment_shear(img, config=self.config),
            AugmentationType.ELASTIC: lambda img: augment_elastic(img, self.config.elastic_alpha, self.config.elastic_sigma),
            AugmentationType.PERSPECTIVE: lambda img: augment_perspective(img),
            AugmentationType.BRIGHTNESS: lambda img: augment_brightness(img, config=self.config),
            AugmentationType.CONTRAST: lambda img: augment_contrast(img, config=self.config),
            AugmentationType.SATURATION: lambda img: augment_saturation(img, config=self.config),
            AugmentationType.HUE_SHIFT: lambda img: augment_hue_shift(img, config=self.config),
            AugmentationType.COLOR_JITTER: lambda img: augment_color_jitter(img, config=self.config),
            AugmentationType.GAUSSIAN_NOISE: lambda img: augment_gaussian_noise(img, config=self.config),
            AugmentationType.SALT_PEPPER: lambda img: augment_salt_pepper(img, config=self.config),
            AugmentationType.SPECKLE: lambda img: augment_speckle_noise(img),
            AugmentationType.POISSON: lambda img: augment_poisson_noise(img),
            AugmentationType.MOTION_BLUR: lambda img: augment_motion_blur(img, config=self.config),
            AugmentationType.PROJECTOR_ARTIFACT: lambda img: augment_projector_artifact(img),
            AugmentationType.LIGHTING_GRADIENT: lambda img: augment_lighting_gradient(img),
            AugmentationType.JPEG_COMPRESSION: lambda img: augment_jpeg_compression(img, config=self.config),
            AugmentationType.RANDOM_ERASING: lambda img: augment_random_erasing(img, config=self.config),
            AugmentationType.CUTOUT: lambda img: augment_cutout(img, config=self.config),
        }

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline with per-step probability."""
        result = image.copy()
        for aug_type in self.config.augmentations:
            if random.random() < self.config.probability:
                fn = self._augment_map.get(aug_type)
                if fn:
                    result = fn(result)
        return result

    def generate_batch(self, image: np.ndarray, count: int = 5) -> List[np.ndarray]:
        """Generate multiple augmented versions of a single image."""
        return [self.augment(image) for _ in range(count)]
