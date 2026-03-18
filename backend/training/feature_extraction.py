"""Feature Extraction Module for ML Training.

Extracts discriminative features from preprocessed images and audio
for use in training classifiers and neural networks.

Vision Features:
    1. HOG  — Histogram of Oriented Gradients (shape/edge features)
    2. LBP  — Local Binary Patterns (texture features)
    3. SIFT — Scale-Invariant Feature Transform (keypoint features)
    4. ORB  — Oriented FAST and Rotated BRIEF (fast keypoints)
    5. Color Histograms — HSV/LAB color distribution
    6. Edge Density Maps — Spatial distribution of edge information
    7. Gabor Filters — Texture at multiple orientations/frequencies
    8. Hu Moments — Shape descriptors (rotation/scale invariant)
    9. Haralick Texture — Gray-level co-occurrence matrix features

Audio Features:
    1. MFCC — Mel-Frequency Cepstral Coefficients
    2. Spectral Features — Centroid, bandwidth, rolloff, flux
    3. Prosodic Features — Pitch, energy, speech rate
    4. Chroma — Pitch class distribution
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FeatureVector:
    """Container for extracted features with metadata."""
    name: str
    values: np.ndarray
    shape: Tuple[int, ...]
    description: str = ""


# ═══════════════════════════════════════════════════════════════
#  VISION FEATURES
# ═══════════════════════════════════════════════════════════════

# ── 1. HOG (Histogram of Oriented Gradients) ─────────────────

def extract_hog(image: np.ndarray, cell_size: int = 8,
                block_size: int = 2, nbins: int = 9,
                resize: Tuple[int, int] = (128, 128)) -> FeatureVector:
    """Extract HOG features — captures shape and edge information.
    
    Process:
    1. Compute gradients (magnitude and direction) at each pixel
    2. Divide image into cells, compute histogram of gradient orientations
    3. Normalize histograms across blocks for illumination invariance
    
    Used for: Professor pose classification, slide region type detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, resize)

    win_size = resize
    block_size_px = (cell_size * block_size, cell_size * block_size)
    block_stride = (cell_size, cell_size)
    cell_size_px = (cell_size, cell_size)

    hog = cv2.HOGDescriptor(
        win_size, block_size_px, block_stride, cell_size_px, nbins
    )
    features = hog.compute(gray)

    return FeatureVector(
        name="HOG",
        values=features.flatten(),
        shape=features.shape,
        description=f"HOG: {len(features.flatten())} dims, {nbins} bins, {cell_size}px cells",
    )


# ── 2. LBP (Local Binary Patterns) ──────────────────────────

def extract_lbp(image: np.ndarray, radius: int = 1,
                n_points: int = 8) -> FeatureVector:
    """Extract Local Binary Pattern features — captures micro-texture.
    
    For each pixel, compares with circular neighbors:
    - Neighbor >= center pixel → 1
    - Neighbor < center pixel → 0
    Concatenate bits → LBP code. Histogram of codes = feature vector.
    
    Uses vectorized numpy for fast computation.
    Used for: Whiteboard vs slide detection, text region identification."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (128, 128)).astype(np.int16)
    h, w = gray.shape

    # Vectorized LBP: precompute all neighbor offsets
    lbp_image = np.zeros((h, w), dtype=np.uint8)
    for k in range(n_points):
        angle = 2 * np.pi * k / n_points
        dy = -radius * np.sin(angle)
        dx = radius * np.cos(angle)
        # Use integer offsets for speed
        iy, ix = int(round(dy)), int(round(dx))
        # Shifted neighbor array vs center array — fully vectorized comparison
        y_lo = max(0, -iy)
        y_hi = min(h, h - iy)
        x_lo = max(0, -ix)
        x_hi = min(w, w - ix)
        center_crop = gray[y_lo:y_hi, x_lo:x_hi]
        neighbor_crop = gray[y_lo + iy:y_hi + iy, x_lo + ix:x_hi + ix]
        lbp_image[y_lo:y_hi, x_lo:x_hi] |= ((neighbor_crop >= center_crop).astype(np.uint8) << k)

    # Compute histogram
    hist, _ = np.histogram(lbp_image, bins=2 ** n_points, range=(0, 2 ** n_points))
    hist = hist.astype(np.float32) / max(hist.sum(), 1)

    return FeatureVector(
        name="LBP",
        values=hist,
        shape=hist.shape,
        description=f"LBP: {len(hist)} bins, radius={radius}, points={n_points}",
    )


# ── 3. SIFT (Scale-Invariant Feature Transform) ─────────────

def extract_sift(image: np.ndarray, max_keypoints: int = 100) -> FeatureVector:
    """Extract SIFT keypoints and descriptors.
    
    Process:
    1. Scale-space extrema detection (DoG pyramid)
    2. Keypoint localization (sub-pixel refinement)
    3. Orientation assignment (gradient histogram)
    4. Descriptor computation (128-d vector per keypoint)
    
    Scale and rotation invariant — used for slide matching across frames."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.zeros((1, 128), dtype=np.float32)

    # Aggregate: mean and std of all descriptors
    mean_desc = np.mean(descriptors, axis=0)
    std_desc = np.std(descriptors, axis=0)
    aggregated = np.concatenate([mean_desc, std_desc])

    return FeatureVector(
        name="SIFT",
        values=aggregated,
        shape=aggregated.shape,
        description=f"SIFT: {len(keypoints)} keypoints, 256-d aggregated",
    )


# ── 4. ORB (Oriented FAST and Rotated BRIEF) ────────────────

def extract_orb(image: np.ndarray, max_keypoints: int = 200) -> FeatureVector:
    """Extract ORB features — fast binary descriptor alternative to SIFT.
    
    FAST corner detection + BRIEF descriptor + orientation.
    Much faster than SIFT, suitable for real-time slide change detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.zeros((1, 32), dtype=np.uint8)

    # Aggregate binary descriptors
    mean_desc = np.mean(descriptors.astype(np.float32), axis=0)

    return FeatureVector(
        name="ORB",
        values=mean_desc,
        shape=mean_desc.shape,
        description=f"ORB: {len(keypoints)} keypoints, 32-d aggregated",
    )


# ── 5. Color Histograms ─────────────────────────────────────

def extract_color_histogram(image: np.ndarray, bins: int = 32,
                             color_space: str = "hsv") -> FeatureVector:
    """Extract color distribution features in HSV or LAB color space.
    
    HSV separates color (H), saturation (S), and brightness (V) —
    more robust to lighting changes than RGB.
    
    Used for: Slide background classification, projector calibration."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        converted = image

    histograms = []
    for channel in range(3):
        hist = cv2.calcHist([converted], [channel], None, [bins], [0, 256])
        hist = hist.flatten().astype(np.float32)
        hist /= max(hist.sum(), 1)
        histograms.append(hist)

    features = np.concatenate(histograms)
    return FeatureVector(
        name=f"ColorHist_{color_space.upper()}",
        values=features,
        shape=features.shape,
        description=f"Color histogram ({color_space}): {len(features)} dims, {bins} bins/channel",
    )


# ── 6. Edge Density Map ─────────────────────────────────────

def extract_edge_density(image: np.ndarray, grid_size: int = 8) -> FeatureVector:
    """Compute edge density in a spatial grid.
    
    Divides image into grid_size×grid_size cells, computes ratio of
    edge pixels in each cell. Creates a spatial map of edge information.
    
    Used for: Detecting text-heavy vs image-heavy slide regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = edges[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            density_map[i, j] = np.sum(cell > 0) / max(cell.size, 1)

    features = density_map.flatten()
    return FeatureVector(
        name="EdgeDensity",
        values=features,
        shape=(grid_size, grid_size),
        description=f"Edge density: {grid_size}x{grid_size} grid = {len(features)} dims",
    )


# ── 7. Gabor Filter Features ────────────────────────────────

def extract_gabor_features(image: np.ndarray,
                            orientations: int = 8,
                            frequencies: List[float] = None) -> FeatureVector:
    """Apply Gabor filter bank and extract texture features.
    
    Gabor filters respond to specific spatial frequencies at specific
    orientations — models how the human visual cortex processes textures.
    
    Used for: Distinguishing text, diagrams, and photos on slides."""
    if frequencies is None:
        frequencies = [0.05, 0.1, 0.2, 0.4]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (128, 128)).astype(np.float32)

    features = []
    for freq in frequencies:
        for theta_idx in range(orientations):
            theta = theta_idx * np.pi / orientations
            kernel = cv2.getGaborKernel(
                ksize=(21, 21), sigma=4.0, theta=theta,
                lambd=1.0 / freq, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Mean and variance of filtered response
            features.extend([np.mean(filtered), np.var(filtered)])

    feature_array = np.array(features, dtype=np.float32)
    return FeatureVector(
        name="Gabor",
        values=feature_array,
        shape=feature_array.shape,
        description=f"Gabor: {len(features)} dims ({orientations} orientations × {len(frequencies)} frequencies × 2)",
    )


# ── 8. Hu Moments ───────────────────────────────────────────

def extract_hu_moments(image: np.ndarray) -> FeatureVector:
    """Extract 7 Hu moments — shape descriptors invariant to
    translation, scale, and rotation.
    
    Used for: Gesture shape classification (pointing, waving, etc.)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform for better numerical properties
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return FeatureVector(
        name="HuMoments",
        values=hu_log.astype(np.float32),
        shape=(7,),
        description="Hu moments: 7 rotation/scale/translation invariant shape descriptors",
    )


# ── 9. Haralick Texture Features (GLCM) ─────────────────────

def extract_haralick(image: np.ndarray, distances: List[int] = None,
                      angles: List[float] = None) -> FeatureVector:
    """Extract Haralick texture features from Gray-Level Co-occurrence Matrix (GLCM).
    
    GLCM captures how often pairs of pixel values occur at a given
    spatial relationship. Haralick features summarize this to:
    contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    
    Uses vectorized numpy for fast computation.
    Used for: Slide background texture classification."""
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (64, 64))  # Smaller size for speed

    # Quantize to 16 levels
    gray_q = (gray // 16).astype(np.uint8)
    levels = 16

    # Precompute index grids for vectorized feature extraction
    i_grid, j_grid = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
    diff_grid = np.abs(i_grid - j_grid)

    all_features = []
    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))

            h, w = gray_q.shape
            # Vectorized GLCM construction using numpy advanced indexing
            y_lo, y_hi = max(0, -dy), min(h, h - dy)
            x_lo, x_hi = max(0, -dx), min(w, w - dx)
            src = gray_q[y_lo:y_hi, x_lo:x_hi].ravel()
            dst = gray_q[y_lo + dy:y_hi + dy, x_lo + dx:x_hi + dx].ravel()

            glcm = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(glcm, (src, dst), 1)

            total = glcm.sum()
            if total > 0:
                glcm /= total

            # Vectorized Haralick features
            contrast = float(np.sum(diff_grid ** 2 * glcm))
            homogeneity = float(np.sum(glcm / (1 + diff_grid)))
            energy = float(np.sum(glcm ** 2))

            mu_i = np.sum(i_grid * glcm)
            mu_j = np.sum(j_grid * glcm)
            sigma_i = np.sqrt(np.sum(((i_grid - mu_i) ** 2) * glcm))
            sigma_j = np.sqrt(np.sum(((j_grid - mu_j) ** 2) * glcm))
            if sigma_i > 0 and sigma_j > 0:
                correlation = float(np.sum(glcm * (i_grid - mu_i) * (j_grid - mu_j) / (sigma_i * sigma_j)))
            else:
                correlation = 0.0

            all_features.extend([contrast, homogeneity, energy, correlation])

    features = np.array(all_features, dtype=np.float32)
    return FeatureVector(
        name="Haralick",
        values=features,
        shape=features.shape,
        description=f"Haralick GLCM: {len(features)} dims ({len(distances)}d × {len(angles)}a × 4 features)",
    )


# ═══════════════════════════════════════════════════════════════
#  AUDIO FEATURES
# ═══════════════════════════════════════════════════════════════

def extract_mfcc(audio: np.ndarray, sr: int = 16000,
                 n_mfcc: int = 13, hop_length: int = 512) -> FeatureVector:
    """Extract Mel-Frequency Cepstral Coefficients (MFCC).
    
    Pipeline:
    1. Pre-emphasis filter
    2. Windowing (Hamming)
    3. FFT → Power spectrum
    4. Mel filterbank → Mel spectrum
    5. Log compression
    6. DCT → Cepstral coefficients
    
    Standard feature for speech/voice analysis.
    Used for: Voice emphasis detection, speaker identification."""
    # Pre-emphasis
    pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    frame_length = int(0.025 * sr)
    n_frames = 1 + (len(pre_emphasized) - frame_length) // hop_length
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = pre_emphasized[start:start + frame_length]

    # Apply Hamming window
    window = np.hamming(frame_length)
    frames *= window

    # FFT
    nfft = 512
    mag_spectrum = np.abs(np.fft.rfft(frames, nfft))
    power_spectrum = mag_spectrum ** 2 / nfft

    # Mel filterbank
    n_filters = 26
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fft_bins = np.floor((nfft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_filters, nfft // 2 + 1))
    for i in range(n_filters):
        for j in range(fft_bins[i], fft_bins[i + 1]):
            filterbank[i, j] = (j - fft_bins[i]) / max(fft_bins[i + 1] - fft_bins[i], 1)
        for j in range(fft_bins[i + 1], fft_bins[i + 2]):
            filterbank[i, j] = (fft_bins[i + 2] - j) / max(fft_bins[i + 2] - fft_bins[i + 1], 1)

    # Apply filterbank
    mel_spectrum = np.dot(power_spectrum, filterbank.T)
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
    log_mel = np.log(mel_spectrum)

    # DCT
    from scipy.fft import dct
    mfccs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # Aggregate: mean and std across frames
    mean_mfcc = np.mean(mfccs, axis=0)
    std_mfcc = np.std(mfccs, axis=0)
    features = np.concatenate([mean_mfcc, std_mfcc])

    return FeatureVector(
        name="MFCC",
        values=features.astype(np.float32),
        shape=features.shape,
        description=f"MFCC: {n_mfcc} coefficients × 2 (mean+std) = {len(features)} dims",
    )


def extract_spectral_features(audio: np.ndarray, sr: int = 16000,
                                hop_length: int = 512) -> FeatureVector:
    """Extract spectral features: centroid, bandwidth, rolloff, flux.
    
    Centroid  — "brightness" of sound (higher = more emphasis)
    Bandwidth — spread of spectrum (wider = more varied speaking)
    Rolloff   — frequency below which 85% of energy lies
    Flux      — rate of spectral change (high during emphasis)"""
    frame_length = int(0.025 * sr)
    n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)

    centroids = []
    bandwidths = []
    rolloffs = []
    fluxes = []
    prev_spectrum = None

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))

        spectrum = np.abs(np.fft.rfft(frame * np.hamming(frame_length)))
        freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)

        total_energy = np.sum(spectrum) + 1e-10

        # Spectral centroid
        centroid = np.sum(freqs * spectrum) / total_energy
        centroids.append(centroid)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_energy)
        bandwidths.append(bandwidth)

        # Spectral rolloff (85%)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        rolloffs.append(rolloff)

        # Spectral flux
        if prev_spectrum is not None:
            flux = np.sum((spectrum - prev_spectrum) ** 2)
            fluxes.append(flux)
        prev_spectrum = spectrum

    features = np.array([
        np.mean(centroids), np.std(centroids),
        np.mean(bandwidths), np.std(bandwidths),
        np.mean(rolloffs), np.std(rolloffs),
        np.mean(fluxes) if fluxes else 0, np.std(fluxes) if fluxes else 0,
    ], dtype=np.float32)

    return FeatureVector(
        name="Spectral",
        values=features,
        shape=features.shape,
        description="Spectral: centroid, bandwidth, rolloff, flux (mean+std) = 8 dims",
    )


def extract_prosodic_features(audio: np.ndarray, sr: int = 16000) -> FeatureVector:
    """Extract prosodic features for emphasis detection:
    
    Pitch (F0) — fundamental frequency (higher → emphasis/excitement)
    Energy     — signal power (louder → emphasis)  
    Speech Rate — zero crossing rate proxy (faster/slower = emphasis)
    Pause Ratio — fraction of silence (pauses before/after important points)"""
    # Energy per frame
    frame_length = int(0.025 * sr)
    hop = frame_length // 2
    n_frames = max(1, (len(audio) - frame_length) // hop)

    energies = []
    zcrs = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        energies.append(np.sum(frame ** 2))
        zcrs.append(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame)))

    energies = np.array(energies)
    zcrs = np.array(zcrs)

    # Silence detection (pause ratio)
    energy_threshold = np.mean(energies) * 0.1
    silent_frames = np.sum(energies < energy_threshold)
    pause_ratio = silent_frames / max(len(energies), 1)

    # Simple pitch estimation using autocorrelation
    min_lag = int(sr / 500)  # 500 Hz max
    max_lag = int(sr / 50)   # 50 Hz min
    pitches = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_length]
        if len(frame) < max_lag:
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        if max_lag < len(corr):
            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0:
                lag = np.argmax(search_region) + min_lag
                if lag > 0:
                    pitches.append(sr / lag)

    features = np.array([
        np.mean(energies), np.std(energies), np.max(energies),
        np.mean(zcrs), np.std(zcrs),
        np.mean(pitches) if pitches else 0,
        np.std(pitches) if pitches else 0,
        np.max(pitches) if pitches else 0,
        pause_ratio,
    ], dtype=np.float32)

    return FeatureVector(
        name="Prosodic",
        values=features,
        shape=features.shape,
        description="Prosodic: energy(3), ZCR(2), pitch(3), pause_ratio = 9 dims",
    )


# ── Combined Feature Extraction ─────────────────────────────

def extract_all_vision_features(image: np.ndarray) -> Dict[str, FeatureVector]:
    """Extract all vision features from an image."""
    return {
        "hog": extract_hog(image),
        "lbp": extract_lbp(image),
        "sift": extract_sift(image),
        "orb": extract_orb(image),
        "color_hsv": extract_color_histogram(image, color_space="hsv"),
        "color_lab": extract_color_histogram(image, color_space="lab"),
        "edge_density": extract_edge_density(image),
        "gabor": extract_gabor_features(image),
        "hu_moments": extract_hu_moments(image),
        "haralick": extract_haralick(image),
    }


def extract_all_audio_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, FeatureVector]:
    """Extract all audio features from a waveform."""
    return {
        "mfcc": extract_mfcc(audio, sr),
        "spectral": extract_spectral_features(audio, sr),
        "prosodic": extract_prosodic_features(audio, sr),
    }


def concatenate_features(features: Dict[str, FeatureVector]) -> np.ndarray:
    """Concatenate all feature vectors into a single flat array for ML training."""
    return np.concatenate([f.values for f in features.values()])
