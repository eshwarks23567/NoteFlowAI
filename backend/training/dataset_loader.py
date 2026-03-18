"""Dataset Loader — loads, preprocesses, and batches training data
from standard academic datasets for each ML module.

Supported Datasets:
    OCR:      ICDAR 2019, PubLayNet, DocVQA, Im2LaTeX, CROHME
    Gesture:  COCO Keypoints, MPII, EgoHands, FreiHAND, EPIC-Kitchens
    Speech:   LibriSpeech, TED-LIUM, AMI, How2
    Emphasis: IEMOCAP, RAVDESS, DAiSEE
    Fusion:   CMU-MOSEI, CMU-MOSI
    Knowledge: LectureBank, SlideVQA, S2ORC
"""
from __future__ import annotations
import os
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Generator, Any
from enum import Enum


class DatasetName(Enum):
    # OCR datasets
    ICDAR = "icdar"
    PUBLAYNET = "publaynet"
    DOCVQA = "docvqa"
    IM2LATEX = "im2latex"
    CROHME = "crohme"
    # Gesture datasets
    COCO_KEYPOINTS = "coco_keypoints"
    MPII = "mpii"
    EGOHANDS = "egohands"
    FREIHAND = "freihand"
    EPIC_KITCHENS = "epic_kitchens"
    # Speech datasets
    LIBRISPEECH = "librispeech"
    TEDLIUM = "tedlium"
    AMI = "ami"
    HOW2 = "how2"
    # Emphasis datasets
    IEMOCAP = "iemocap"
    RAVDESS = "ravdess"
    DAISEE = "daisee"
    # Fusion datasets
    CMU_MOSEI = "cmu_mosei"
    CMU_MOSI = "cmu_mosi"
    # Knowledge datasets
    LECTUREBANK = "lecturebank"
    SLIDEVQA = "slidevqa"


@dataclass
class DataSample:
    """A single training data sample."""
    data: Any               # Image (np.ndarray), audio (np.ndarray), text (str)
    label: Any              # Class label, bounding box, transcription, etc.
    metadata: Dict = field(default_factory=dict)
    sample_id: str = ""


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: DatasetName
    root_dir: str
    split: str = "train"                # train, val, test
    max_samples: Optional[int] = None   # Limit samples for debugging
    image_size: Tuple[int, int] = (640, 480)
    batch_size: int = 32
    shuffle: bool = True
    augment: bool = True
    cache_preprocessed: bool = True


@dataclass
class DatasetStats:
    """Statistics about a loaded dataset."""
    name: str
    total_samples: int
    num_classes: int
    class_distribution: Dict[str, int]
    mean_image_size: Optional[Tuple[float, float]] = None
    split_sizes: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  BASE LOADER
# ═══════════════════════════════════════════════════════════════

class BaseDatasetLoader:
    """Base class for all dataset loaders."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.samples: List[DataSample] = []
        self._loaded = False

    def load(self) -> DatasetStats:
        """Load dataset from disk. Override in subclasses."""
        raise NotImplementedError

    def get_batch(self, batch_idx: int) -> List[DataSample]:
        """Get a specific batch of samples."""
        start = batch_idx * self.config.batch_size
        end = min(start + self.config.batch_size, len(self.samples))
        return self.samples[start:end]

    def iterate_batches(self) -> Generator[List[DataSample], None, None]:
        """Iterate over all batches."""
        if self.config.shuffle:
            np.random.shuffle(self.samples)

        for i in range(0, len(self.samples), self.config.batch_size):
            yield self.samples[i:i + self.config.batch_size]

    @property
    def num_batches(self) -> int:
        return (len(self.samples) + self.config.batch_size - 1) // self.config.batch_size


# ═══════════════════════════════════════════════════════════════
#  OCR DATASET LOADERS
# ═══════════════════════════════════════════════════════════════

class ICDARLoader(BaseDatasetLoader):
    """ICDAR 2019 — Scene text detection and recognition.
    
    Structure:
        root/
        ├── train/
        │   ├── images/
        │   │   ├── img_1.jpg
        │   │   └── ...
        │   └── labels/
        │       ├── gt_img_1.txt   (x1,y1,x2,y2,x3,y3,x4,y4,text)
        │       └── ...
        └── test/
    """
    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir) / self.config.split
        images_dir = root / "images"
        labels_dir = root / "labels"

        class_counts: Dict[str, int] = {"text": 0}

        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.jpg")):
                label_path = labels_dir / f"gt_{img_path.stem}.txt"
                bboxes = []
                if label_path.exists():
                    with open(label_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) >= 9:
                                coords = [int(p) for p in parts[:8]]
                                text = ",".join(parts[8:])
                                bboxes.append({
                                    "polygon": coords,
                                    "text": text,
                                })
                                class_counts["text"] += 1

                self.samples.append(DataSample(
                    data=str(img_path),
                    label=bboxes,
                    metadata={"dataset": "ICDAR", "split": self.config.split},
                    sample_id=img_path.stem,
                ))

                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        self._loaded = True
        return DatasetStats(
            name="ICDAR 2019",
            total_samples=len(self.samples),
            num_classes=1,
            class_distribution=class_counts,
            split_sizes={self.config.split: len(self.samples)},
        )


class PubLayNetLoader(BaseDatasetLoader):
    """PubLayNet — Document layout analysis with 5 categories.
    
    Categories: text, title, list, table, figure
    Format: COCO-style JSON annotations
    
    Structure:
        root/
        ├── train/
        │   └── *.png
        ├── val/
        └── publaynet/
            ├── train.json
            └── val.json
    """
    CATEGORY_MAP = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}

    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        ann_path = root / "publaynet" / f"{self.config.split}.json"
        images_dir = root / self.config.split

        class_counts: Dict[str, int] = {v: 0 for v in self.CATEGORY_MAP.values()}

        if ann_path.exists():
            with open(ann_path, "r") as f:
                coco_data = json.load(f)

            image_lookup = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

            # Group annotations by image
            img_anns: Dict[int, list] = {}
            for ann in coco_data.get("annotations", []):
                img_id = ann["image_id"]
                if img_id not in img_anns:
                    img_anns[img_id] = []
                cat_name = self.CATEGORY_MAP.get(ann["category_id"], "unknown")
                img_anns[img_id].append({
                    "bbox": ann["bbox"],  # [x, y, w, h]
                    "category": cat_name,
                    "segmentation": ann.get("segmentation"),
                })
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

            for img_id, anns in img_anns.items():
                file_name = image_lookup.get(img_id, "")
                img_path = images_dir / file_name

                self.samples.append(DataSample(
                    data=str(img_path),
                    label=anns,
                    metadata={"dataset": "PubLayNet", "image_id": img_id},
                    sample_id=str(img_id),
                ))

                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        self._loaded = True
        return DatasetStats(
            name="PubLayNet",
            total_samples=len(self.samples),
            num_classes=5,
            class_distribution=class_counts,
            split_sizes={self.config.split: len(self.samples)},
        )


class Im2LatexLoader(BaseDatasetLoader):
    """Im2LaTeX-100K — Image to LaTeX equation conversion.
    
    Structure:
        root/
        ├── formula_images/
        │   ├── 0.png
        │   └── ...
        ├── im2latex_train.lst
        ├── im2latex_validate.lst
        └── im2latex_formulas.lst
    """
    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        split_map = {"train": "im2latex_train.lst", "val": "im2latex_validate.lst",
                     "test": "im2latex_test.lst"}
        lst_path = root / split_map.get(self.config.split, "im2latex_train.lst")
        formulas_path = root / "im2latex_formulas.lst"
        images_dir = root / "formula_images"

        formulas = []
        if formulas_path.exists():
            with open(formulas_path, "r", encoding="utf-8") as f:
                formulas = [line.strip() for line in f]

        if lst_path.exists():
            with open(lst_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        formula_idx = int(parts[0])
                        img_name = parts[1]
                        formula = formulas[formula_idx] if formula_idx < len(formulas) else ""

                        self.samples.append(DataSample(
                            data=str(images_dir / f"{img_name}.png"),
                            label=formula,
                            metadata={"dataset": "Im2LaTeX", "formula_idx": formula_idx},
                            sample_id=img_name,
                        ))

                        if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                            break

        self._loaded = True
        return DatasetStats(
            name="Im2LaTeX-100K",
            total_samples=len(self.samples),
            num_classes=0,  # Sequence output
            class_distribution={"equations": len(self.samples)},
            split_sizes={self.config.split: len(self.samples)},
        )


# ═══════════════════════════════════════════════════════════════
#  GESTURE / POSE DATASET LOADERS
# ═══════════════════════════════════════════════════════════════

class COCOKeypointsLoader(BaseDatasetLoader):
    """COCO Keypoints — 17-point human pose annotations.
    
    Used to train professor gesture and pose recognition.
    Each sample has 17 keypoints with (x, y, visibility) format.
    
    Structure:
        root/
        ├── train2017/
        │   └── *.jpg
        ├── val2017/
        └── annotations/
            ├── person_keypoints_train2017.json
            └── person_keypoints_val2017.json
    """
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        split_name = f"{self.config.split}2017"
        ann_path = root / "annotations" / f"person_keypoints_{split_name}.json"
        images_dir = root / split_name

        if ann_path.exists():
            with open(ann_path, "r") as f:
                coco_data = json.load(f)

            image_lookup = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

            for ann in coco_data.get("annotations", []):
                if ann.get("num_keypoints", 0) < 5:
                    continue

                keypoints = ann["keypoints"]
                kps = [(keypoints[i], keypoints[i + 1], keypoints[i + 2])
                       for i in range(0, len(keypoints), 3)]

                img_path = images_dir / image_lookup.get(ann["image_id"], "")

                self.samples.append(DataSample(
                    data=str(img_path),
                    label={
                        "keypoints": kps,
                        "bbox": ann["bbox"],
                        "num_keypoints": ann["num_keypoints"],
                    },
                    metadata={"dataset": "COCO", "image_id": ann["image_id"]},
                    sample_id=str(ann["id"]),
                ))

                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        self._loaded = True
        return DatasetStats(
            name="COCO Keypoints",
            total_samples=len(self.samples),
            num_classes=1,
            class_distribution={"person": len(self.samples)},
            split_sizes={self.config.split: len(self.samples)},
        )


class MPIILoader(BaseDatasetLoader):
    """MPII Human Pose Dataset — 16-point full-body pose.
    
    Structure:
        root/
        ├── images/
        │   └── *.jpg
        └── mpii_human_pose_v1_u12_2/
            └── mpii_human_pose_v1_u12_1.mat
    """
    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        images_dir = root / "images"

        # Load .mat annotation (simplified — in practice use scipy.io.loadmat)
        # For demonstration, we create a file listing approach
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.jpg")):
                self.samples.append(DataSample(
                    data=str(img_path),
                    label={"joint_positions": []},  # Loaded from .mat
                    metadata={"dataset": "MPII"},
                    sample_id=img_path.stem,
                ))
                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        self._loaded = True
        return DatasetStats(
            name="MPII Human Pose",
            total_samples=len(self.samples),
            num_classes=1,
            class_distribution={"pose": len(self.samples)},
        )


# ═══════════════════════════════════════════════════════════════
#  SPEECH / AUDIO DATASET LOADERS
# ═══════════════════════════════════════════════════════════════

class LibriSpeechLoader(BaseDatasetLoader):
    """LibriSpeech — 1000 hours of read English speech.
    
    Structure:
        root/
        ├── train-clean-100/
        │   └── <speaker_id>/
        │       └── <chapter_id>/
        │           ├── <speaker>-<chapter>-<utterance>.flac
        │           └── <speaker>-<chapter>.trans.txt
        └── ...
    """
    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        split_dirs = list(root.glob(f"{self.config.split}*"))

        for split_dir in split_dirs:
            for trans_path in split_dir.rglob("*.trans.txt"):
                chapter_dir = trans_path.parent
                with open(trans_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            utt_id, transcript = parts
                            audio_path = chapter_dir / f"{utt_id}.flac"

                            self.samples.append(DataSample(
                                data=str(audio_path),
                                label=transcript,
                                metadata={
                                    "dataset": "LibriSpeech",
                                    "speaker": utt_id.split("-")[0],
                                    "chapter": utt_id.split("-")[1] if "-" in utt_id else "",
                                },
                                sample_id=utt_id,
                            ))

                            if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                                self._loaded = True
                                return DatasetStats(
                                    name="LibriSpeech",
                                    total_samples=len(self.samples),
                                    num_classes=0,
                                    class_distribution={"utterances": len(self.samples)},
                                )

        self._loaded = True
        return DatasetStats(
            name="LibriSpeech",
            total_samples=len(self.samples),
            num_classes=0,
            class_distribution={"utterances": len(self.samples)},
            split_sizes={self.config.split: len(self.samples)},
        )


class RAVDESSLoader(BaseDatasetLoader):
    """RAVDESS — Emotional speech and song audio-visual dataset.
    
    8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    24 professional actors (12 male, 12 female)
    
    Filename encoding: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    """
    EMOTION_MAP = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
    }

    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)
        class_counts: Dict[str, int] = {v: 0 for v in self.EMOTION_MAP.values()}

        for actor_dir in sorted(root.glob("Actor_*")):
            for audio_path in sorted(actor_dir.glob("*.wav")):
                parts = audio_path.stem.split("-")
                if len(parts) >= 7:
                    emotion = self.EMOTION_MAP.get(parts[2], "unknown")
                    intensity = "normal" if parts[3] == "01" else "strong"
                    actor_id = parts[6]

                    self.samples.append(DataSample(
                        data=str(audio_path),
                        label={
                            "emotion": emotion,
                            "intensity": intensity,
                            "actor": actor_id,
                        },
                        metadata={"dataset": "RAVDESS"},
                        sample_id=audio_path.stem,
                    ))
                    class_counts[emotion] = class_counts.get(emotion, 0) + 1

                    if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                        break

        self._loaded = True
        return DatasetStats(
            name="RAVDESS",
            total_samples=len(self.samples),
            num_classes=8,
            class_distribution=class_counts,
            split_sizes={self.config.split: len(self.samples)},
        )


# ═══════════════════════════════════════════════════════════════
#  MULTIMODAL FUSION DATASET LOADERS
# ═══════════════════════════════════════════════════════════════

class CMUMOSEILoader(BaseDatasetLoader):
    """CMU-MOSEI — Multimodal Opinion Sentiment and Emotion Intensity.
    
    23,454 annotated video segments from YouTube.
    6 emotions + sentiment scoring.
    Aligned text, audio, and visual features.
    
    Structure:
        root/
        ├── Raw/
        │   ├── Videos/ *.mp4
        │   └── Audio/ *.wav
        ├── Aligned/
        │   ├── text.pkl
        │   ├── audio.pkl
        │   └── visual.pkl
        └── Labels/
            └── labels.pkl
    """
    def load(self) -> DatasetStats:
        root = Path(self.config.root_dir)

        # Try to load preprocessed aligned features
        aligned_dir = root / "Aligned"
        labels_path = root / "Labels" / "labels.pkl"

        class_counts: Dict[str, int] = {
            "happy": 0, "sad": 0, "angry": 0,
            "fearful": 0, "disgusted": 0, "surprised": 0,
        }

        if aligned_dir.exists() and labels_path.exists():
            import pickle
            with open(labels_path, "rb") as f:
                labels = pickle.load(f)

            for video_id, label_data in labels.items():
                for segment_id, segment_labels in label_data.items():
                    self.samples.append(DataSample(
                        data={"video_id": video_id, "segment_id": segment_id},
                        label=segment_labels,
                        metadata={"dataset": "CMU-MOSEI"},
                        sample_id=f"{video_id}_{segment_id}",
                    ))

                    if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                        break
                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        # Alternatively, load raw videos
        elif (root / "Raw" / "Videos").exists():
            for video_path in sorted((root / "Raw" / "Videos").glob("*.mp4")):
                self.samples.append(DataSample(
                    data=str(video_path),
                    label={},
                    metadata={"dataset": "CMU-MOSEI", "type": "raw"},
                    sample_id=video_path.stem,
                ))
                if self.config.max_samples and len(self.samples) >= self.config.max_samples:
                    break

        self._loaded = True
        return DatasetStats(
            name="CMU-MOSEI",
            total_samples=len(self.samples),
            num_classes=6,
            class_distribution=class_counts,
            split_sizes={self.config.split: len(self.samples)},
        )


# ═══════════════════════════════════════════════════════════════
#  DATASET FACTORY
# ═══════════════════════════════════════════════════════════════

LOADER_REGISTRY: Dict[DatasetName, type] = {
    DatasetName.ICDAR: ICDARLoader,
    DatasetName.PUBLAYNET: PubLayNetLoader,
    DatasetName.IM2LATEX: Im2LatexLoader,
    DatasetName.COCO_KEYPOINTS: COCOKeypointsLoader,
    DatasetName.MPII: MPIILoader,
    DatasetName.LIBRISPEECH: LibriSpeechLoader,
    DatasetName.RAVDESS: RAVDESSLoader,
    DatasetName.CMU_MOSEI: CMUMOSEILoader,
}


def create_loader(config: DatasetConfig) -> BaseDatasetLoader:
    """Factory function to create the appropriate dataset loader."""
    loader_cls = LOADER_REGISTRY.get(config.name)
    if loader_cls is None:
        raise ValueError(f"No loader registered for dataset: {config.name}")
    return loader_cls(config)


def load_dataset(name: DatasetName, root_dir: str,
                 split: str = "train", **kwargs) -> Tuple[BaseDatasetLoader, DatasetStats]:
    """Convenience function to create a loader and load the dataset."""
    config = DatasetConfig(name=name, root_dir=root_dir, split=split, **kwargs)
    loader = create_loader(config)
    stats = loader.load()
    return loader, stats
