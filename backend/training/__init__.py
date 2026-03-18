"""ML Training Module for NoteFlow AI — Live Lecture Note-Taker.

This package contains the complete training pipeline for all ML
components of the live lecture note-taking system.

Modules:
    preprocessing       — 15+ image filtering techniques
    segmentation        — 9 image segmentation methods
    augmentation        — 25+ data augmentation techniques
    feature_extraction  — 12 feature extractors (vision + audio)
    dataset_loader      — 8 dataset loaders with factory pattern

Training Pipelines:
    train_slide_ocr     — CNN-based slide text detection + CRNN recognizer
    train_gesture       — LSTM-based professor gesture classifier
    train_emphasis      — 1D-CNN + BiLSTM voice emphasis detector
    train_fusion        — Cross-attention multimodal importance fusion
"""

__version__ = "1.0.0"

# Core modules
from training.preprocessing import ImagePreprocessor, PreprocessConfig, FilterType
from training.segmentation import (
    SegmentedRegion,
    segment_contours,
    segment_watershed,
    segment_grabcut,
    detect_text_regions_mser,
    segment_slide_layout,
)
from training.augmentation import DataAugmentor, AugmentConfig, AugmentationType
from training.feature_extraction import (
    FeatureVector,
    extract_hog,
    extract_lbp,
    extract_sift,
    extract_mfcc,
    extract_all_vision_features,
    extract_all_audio_features,
)
from training.dataset_loader import (
    DatasetName,
    DataSample,
    DatasetConfig,
    create_loader,
    load_dataset,
)

# Training pipelines
from training.train_slide_ocr import OCRTrainingPipeline, OCRTrainingConfig
from training.train_gesture import GestureTrainingPipeline, GestureTrainingConfig
from training.train_emphasis import EmphasisTrainingPipeline, EmphasisTrainingConfig
from training.train_fusion import FusionTrainingPipeline, FusionTrainingConfig

__all__ = [
    # Preprocessing
    "ImagePreprocessor", "PreprocessConfig", "FilterType",
    # Segmentation
    "SegmentedRegion", "segment_contours", "segment_watershed",
    "segment_grabcut", "detect_text_regions_mser", "segment_slide_layout",
    # Augmentation
    "DataAugmentor", "AugmentConfig", "AugmentationType",
    # Feature Extraction
    "FeatureVector", "extract_hog", "extract_lbp", "extract_sift",
    "extract_mfcc", "extract_all_vision_features", "extract_all_audio_features",
    # Dataset Loader
    "DatasetName", "DataSample", "DatasetConfig", "create_loader", "load_dataset",
    # Training Pipelines
    "OCRTrainingPipeline", "OCRTrainingConfig",
    "GestureTrainingPipeline", "GestureTrainingConfig",
    "EmphasisTrainingPipeline", "EmphasisTrainingConfig",
    "FusionTrainingPipeline", "FusionTrainingConfig",
]
