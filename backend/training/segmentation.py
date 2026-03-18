"""Image Segmentation Techniques for Lecture Content Analysis.

Implements various segmentation methods to isolate regions of interest
in lecture slides and professor video frames.

Techniques:
    1. Thresholding     — Global, Adaptive, Otsu's, multi-Otsu
    2. Contour-based    — Find, filter, and extract contour regions
    3. Watershed        — Marker-based watershed segmentation
    4. GrabCut          — Interactive foreground extraction
    5. Color-space      — HSV/LAB range segmentation
    6. Connected Components — Label and extract connected regions
    7. Region Growing   — Seed-based region expansion
    8. Superpixels      — SLIC superpixel oversegmentation
    9. Text Region Detection — MSER + morphological text region isolation
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class SegmentedRegion:
    """A segmented region in an image."""
    mask: np.ndarray           # Binary mask of the region
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int                  # Pixel area
    label: str = ""           # Region classification
    confidence: float = 0.0   # Segmentation confidence


# ── 1. Thresholding Techniques ───────────────────────────────────

def threshold_global(image: np.ndarray, threshold: int = 128,
                     max_val: int = 255) -> np.ndarray:
    """Simple global thresholding — pixels above threshold → white."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Otsu's method — automatically determines optimal threshold
    by minimizing intra-class variance. Assumes bimodal histogram.
    Ideal for separating text from slide background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    threshold_value, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, float(threshold_value)


def threshold_adaptive(image: np.ndarray, block_size: int = 11,
                       c: int = 2, method: str = "gaussian") -> np.ndarray:
    """Adaptive thresholding — threshold varies across image based on local
    neighborhood. Handles variable lighting across projected slides.
    
    Methods:
        gaussian — weighted sum using Gaussian window (smoother)
        mean     — simple mean of neighborhood (sharper transitions)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    adaptive_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                       else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
    )


def threshold_multi_otsu(image: np.ndarray, levels: int = 3) -> Tuple[np.ndarray, List[float]]:
    """Multi-level Otsu thresholding — separates image into N classes.
    Useful for slides with background, text, and diagram colors.
    
    Implementation uses histogram-based approach when cv2 multi-Otsu
    is not available."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()

    # Find optimal thresholds using brute-force for 3 levels
    best_variance = 0
    best_thresholds = [85, 170]

    if levels == 3:
        for t1 in range(1, 255):
            for t2 in range(t1 + 1, 256):
                w0 = hist[:t1].sum()
                w1 = hist[t1:t2].sum()
                w2 = hist[t2:].sum()
                if w0 == 0 or w1 == 0 or w2 == 0:
                    continue
                m0 = np.sum(np.arange(t1) * hist[:t1]) / w0
                m1 = np.sum(np.arange(t1, t2) * hist[t1:t2]) / w1
                m2 = np.sum(np.arange(t2, 256) * hist[t2:]) / w2
                var = w0 * w1 * (m0 - m1) ** 2 + w1 * w2 * (m1 - m2) ** 2 + w0 * w2 * (m0 - m2) ** 2
                if var > best_variance:
                    best_variance = var
                    best_thresholds = [t1, t2]

    # Apply multi-level thresholds
    result = np.zeros_like(gray)
    result[gray >= best_thresholds[1]] = 255
    result[(gray >= best_thresholds[0]) & (gray < best_thresholds[1])] = 128

    return result, [float(t) for t in best_thresholds]


# ── 2. Contour-based Segmentation ────────────────────────────────

def segment_contours(image: np.ndarray, min_area: int = 100,
                     max_area: Optional[int] = None) -> List[SegmentedRegion]:
    """Extract contours and filter by area. Returns segmented regions.
    Used to isolate text blocks, diagrams, and equations on slides."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Adaptive threshold for contour detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = image.shape[0] * image.shape[1]
    if max_area is None:
        max_area = int(img_area * 0.9)

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            regions.append(SegmentedRegion(
                mask=mask,
                bbox=(x, y, w, h),
                area=int(area),
                confidence=min(1.0, area / 1000),
            ))

    # Sort by vertical position (top to bottom) for reading order
    regions.sort(key=lambda r: r.bbox[1])
    return regions


def classify_contour_region(contour: np.ndarray, image_shape: Tuple[int, ...]) -> str:
    """Classify a contour as text, diagram, equation, or header based
    on geometric properties like aspect ratio, solidity, and position."""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(h, 1)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / max(hull_area, 1)
    img_h, img_w = image_shape[:2]
    rel_y = y / img_h  # Relative vertical position

    if aspect_ratio > 5 and h < img_h * 0.1:
        return "text_line"
    elif rel_y < 0.15 and w > img_w * 0.3:
        return "header"
    elif 0.7 < aspect_ratio < 1.4 and solidity > 0.6:
        return "diagram"
    elif solidity < 0.4:
        return "equation"
    elif aspect_ratio > 2:
        return "text_block"
    else:
        return "unknown"


# ── 3. Watershed Segmentation ───────────────────────────────────

def segment_watershed(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """Marker-based watershed segmentation.
    
    Process:
    1. Convert to grayscale → threshold (Otsu)
    2. Morphological opening to remove noise
    3. Distance transform to find sure foreground
    4. Marker labeling from sure foreground regions
    5. Watershed algorithm fills from markers
    
    Used to separate overlapping text/diagram regions on busy slides.
    Returns: (labeled_image, num_segments)"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background — dilation expands the foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground — distance transform + threshold
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background = 1, not 0
    markers[unknown == 255] = 0  # Mark unknown as 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    # markers == -1 are boundaries
    num_segments = num_labels  # excluding background

    return markers, num_segments


# ── 4. GrabCut Segmentation ─────────────────────────────────────

def segment_grabcut(image: np.ndarray, rect: Optional[Tuple[int, int, int, int]] = None,
                    iterations: int = 5) -> np.ndarray:
    """GrabCut foreground extraction — separates foreground (professor, slide)
    from background using iterative graph-cut optimization.
    
    If rect not provided, uses center 60% of image as initial foreground guess."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if rect is None:
        h, w = image.shape[:2]
        rect = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Probable + definite foreground
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    return fg_mask


# ── 5. Color-Space Segmentation ─────────────────────────────────

def segment_by_color_hsv(image: np.ndarray,
                         lower_hsv: Tuple[int, int, int],
                         upper_hsv: Tuple[int, int, int]) -> np.ndarray:
    """Segment regions by HSV color range.
    Useful for detecting:
    - Laser pointer spots (red/green)
    - Whiteboard markers (specific colors)
    - Highlighted text regions on slides"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_skin_region(image: np.ndarray) -> np.ndarray:
    """Detect skin regions using HSV + YCrCb dual color-space approach.
    Used to isolate professor's hands for gesture analysis.
    Combines two color models for robustness across skin tones."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # HSV skin range
    hsv_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))

    # YCrCb skin range
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))

    # Combine both masks
    combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return combined


# ── 6. Connected Components ─────────────────────────────────────

def segment_connected_components(image: np.ndarray,
                                  min_area: int = 50) -> List[SegmentedRegion]:
    """Label connected components and extract as individual regions.
    Faster than contour-based for well-separated text characters/symbols."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    regions = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        mask = (labels == i).astype(np.uint8) * 255
        regions.append(SegmentedRegion(
            mask=mask,
            bbox=(x, y, w, h),
            area=int(area),
            label=f"component_{i}",
            confidence=min(1.0, area / 500),
        ))

    return regions


# ── 7. Region Growing ───────────────────────────────────────────

def segment_region_growing(image: np.ndarray,
                           seed_point: Tuple[int, int],
                           threshold: int = 15) -> np.ndarray:
    """Seed-based region growing — starts from a seed pixel and expands
    to neighboring pixels within intensity threshold.
    
    Useful for isolating specific diagram regions the professor points at."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    seed_value = int(gray[seed_point[1], seed_point[0]])
    stack = [seed_point]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[y, x]:
            continue
        visited[y, x] = 1

        pixel_value = int(gray[y, x])
        if abs(pixel_value - seed_value) <= threshold:
            mask[y, x] = 255
            # 4-connectivity neighbors
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    return mask


# ── 8. Superpixel Segmentation (SLIC) ───────────────────────────

def segment_superpixels(image: np.ndarray, num_segments: int = 200,
                        compactness: float = 10.0) -> Tuple[np.ndarray, int]:
    """SLIC (Simple Linear Iterative Clustering) superpixel segmentation.
    Over-segments image into roughly uniform regions.
    
    Used as preprocessing for:
    - Slide layout analysis (group superpixels → text/diagram/background)
    - Fast region labeling for training data generation"""
    slic = cv2.ximgproc.createSuperpixelSLIC(
        image, algorithm=cv2.ximgproc.SLIC, region_size=int(np.sqrt(image.shape[0] * image.shape[1] / num_segments)),
        ruler=compactness
    )
    slic.iterate(10)
    slic.enforceLabelConnectivity()

    labels = slic.getLabels()
    n_segments = slic.getNumberOfSuperpixels()
    return labels, n_segments


# ── 9. Text Region Detection (MSER) ─────────────────────────────

def detect_text_regions_mser(image: np.ndarray) -> List[SegmentedRegion]:
    """Maximally Stable Extremal Regions (MSER) — detects regions with
    stable intensity that are characteristic of text characters.
    
    Pipeline:
    1. MSER detection → candidate regions
    2. Filter by aspect ratio, size, and fill ratio
    3. Group nearby regions into text lines
    4. Return text region bounding boxes"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    mser = cv2.MSER_create(
        delta=5,
        min_area=60,
        max_area=14400,
        max_variation=0.25,
    )

    regions_raw, _ = mser.detectRegions(gray)
    text_regions = []

    for region in regions_raw:
        x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / max(h, 1)
        area = w * h
        fill_ratio = len(region) / max(area, 1)

        # Text character heuristics
        if 0.1 < aspect_ratio < 10 and 0.2 < fill_ratio < 0.9:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)

            text_regions.append(SegmentedRegion(
                mask=mask,
                bbox=(x, y, w, h),
                area=int(len(region)),
                label="text_char",
                confidence=fill_ratio,
            ))

    # Group nearby characters into text lines
    return _group_into_text_lines(text_regions)


def _group_into_text_lines(regions: List[SegmentedRegion],
                            y_tolerance: int = 15,
                            x_gap_max: int = 50) -> List[SegmentedRegion]:
    """Group individual character regions into text line regions."""
    if not regions:
        return []

    # Sort by y then x
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

    lines: List[List[SegmentedRegion]] = []
    current_line = [regions[0]]

    for region in regions[1:]:
        prev = current_line[-1]
        prev_y_center = prev.bbox[1] + prev.bbox[3] // 2
        curr_y_center = region.bbox[1] + region.bbox[3] // 2

        if abs(curr_y_center - prev_y_center) < y_tolerance:
            current_line.append(region)
        else:
            lines.append(current_line)
            current_line = [region]
    lines.append(current_line)

    # Merge each line into a single region
    merged = []
    for line in lines:
        if not line:
            continue
        x_min = min(r.bbox[0] for r in line)
        y_min = min(r.bbox[1] for r in line)
        x_max = max(r.bbox[0] + r.bbox[2] for r in line)
        y_max = max(r.bbox[1] + r.bbox[3] for r in line)

        total_area = sum(r.area for r in line)
        avg_confidence = np.mean([r.confidence for r in line])

        # Create merged mask
        h_img = line[0].mask.shape[0]
        w_img = line[0].mask.shape[1]
        merged_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for r in line:
            merged_mask = cv2.bitwise_or(merged_mask, r.mask)

        merged.append(SegmentedRegion(
            mask=merged_mask,
            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            area=total_area,
            label="text_line",
            confidence=float(avg_confidence),
        ))

    return merged


# ── Slide Layout Segmentation ───────────────────────────────────

def segment_slide_layout(image: np.ndarray) -> dict:
    """Full slide layout analysis — identifies and labels all regions.
    
    Returns dict with keys:
        header_regions  — Title/header text areas
        body_regions    — Body text areas
        diagram_regions — Charts, graphs, images
        equation_regions — Mathematical equations
        background      — Background mask
    """
    regions = segment_contours(image, min_area=200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    layout = {
        "header_regions": [],
        "body_regions": [],
        "diagram_regions": [],
        "equation_regions": [],
        "background": np.ones(gray.shape, dtype=np.uint8) * 255,
    }

    for region in regions:
        # Find the corresponding contour
        contours, _ = cv2.findContours(
            region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        label = classify_contour_region(contours[0], image.shape)
        region.label = label

        if label == "header":
            layout["header_regions"].append(region)
        elif label in ("text_line", "text_block"):
            layout["body_regions"].append(region)
        elif label == "diagram":
            layout["diagram_regions"].append(region)
        elif label == "equation":
            layout["equation_regions"].append(region)

        # Remove from background
        layout["background"] = cv2.bitwise_and(
            layout["background"],
            cv2.bitwise_not(region.mask)
        )

    return layout
