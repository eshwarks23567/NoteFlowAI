"""Importance Fusion Engine — combines multi-modal signals into a single importance score."""
from __future__ import annotations
from models.schemas import (
    ImportanceEvent, ImportanceLevel, GestureEvent, VoiceEmphasis,
    TranscriptionEvent, EventType
)
import time


# Weights for fusion scoring
WEIGHTS = {
    "gesture":  0.30,
    "voice":    0.25,
    "dwell":    0.20,
    "keyword":  0.15,
    "question": 0.10,
}

# Phrase markers that indicate professor emphasis
EMPHASIS_PHRASES = [
    "this is important",
    "remember that",
    "key takeaway",
    "pay attention",
    "make sure you understand",
    "this will be on the exam",
    "crucial point",
    "the main idea",
    "don't forget",
    "let me repeat",
    "most common mistake",
    "fundamentally",
    "the backbone of",
    "critical concept",
    "essential",
]


def classify_importance(score: float) -> ImportanceLevel:
    if score > 0.7:
        return ImportanceLevel.CRITICAL
    elif score >= 0.4:
        return ImportanceLevel.IMPORTANT
    return ImportanceLevel.SUPPORTING


def compute_keyword_score(text: str) -> float:
    """Score based on emphasis phrase presence in transcript text."""
    text_lower = text.lower()
    matches = sum(1 for phrase in EMPHASIS_PHRASES if phrase in text_lower)
    return min(1.0, matches * 0.5)


def compute_question_score(transcript: list[TranscriptionEvent], window_seconds: float = 30.0) -> float:
    """Score based on recent student questions."""
    now = time.time()
    recent_questions = [
        t for t in transcript
        if t.speaker.value == "student" and (now - t.timestamp) < window_seconds
    ]
    return min(1.0, len(recent_questions) * 0.4)


def fuse_scores(
    gesture_intensity: float = 0.0,
    voice_emphasis: float = 0.0,
    slide_dwell_time: float = 0.0,
    keyword_score: float = 0.0,
    question_score: float = 0.0,
    lecture_time: str = "00:00:00",
    trigger_text: str = "",
) -> ImportanceEvent:
    """Compute fused importance score from all modality signals."""
    score = (
        WEIGHTS["gesture"]  * min(1.0, gesture_intensity) +
        WEIGHTS["voice"]    * min(1.0, voice_emphasis) +
        WEIGHTS["dwell"]    * min(1.0, slide_dwell_time) +
        WEIGHTS["keyword"]  * min(1.0, keyword_score) +
        WEIGHTS["question"] * min(1.0, question_score)
    )
    return ImportanceEvent(
        score=round(score, 3),
        level=classify_importance(score),
        gesture_component=round(gesture_intensity, 3),
        voice_component=round(voice_emphasis, 3),
        dwell_component=round(slide_dwell_time, 3),
        keyword_component=round(keyword_score, 3),
        question_component=round(question_score, 3),
        lecture_time=lecture_time,
        trigger_text=trigger_text,
    )
