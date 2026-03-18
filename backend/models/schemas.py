"""Pydantic models for the Live Lecture Note-Taker."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import time, uuid


# ── Enums ────────────────────────────────────────────────────────
class ImportanceLevel(str, Enum):
    CRITICAL = "critical"       # > 0.7
    IMPORTANT = "important"     # 0.4–0.7
    SUPPORTING = "supporting"   # < 0.4

class SpeakerRole(str, Enum):
    PROFESSOR = "professor"
    STUDENT = "student"
    UNKNOWN = "unknown"

class GestureType(str, Enum):
    POINTING = "pointing"
    SWEEPING = "sweeping"
    COUNTING = "counting"
    EMPHASIS = "emphasis"
    HANDS_RAISED = "hands_raised"
    LEANING_FORWARD = "leaning_forward"
    NONE = "none"

class EventType(str, Enum):
    TRANSCRIPTION = "transcription"
    SLIDE_CHANGE = "slide_change"
    GESTURE = "gesture"
    IMPORTANCE = "importance"
    KEY_CONCEPT = "key_concept"
    ALERT = "alert"
    SESSION_STATUS = "session_status"
    SUMMARY_UPDATE = "summary_update"


# ── Core Models ──────────────────────────────────────────────────
class TranscriptionEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    text: str = ""
    speaker: SpeakerRole = SpeakerRole.PROFESSOR
    confidence: float = 0.0
    is_emphasis_phrase: bool = False
    keywords: list[str] = []

class SlideEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    slide_number: int = 0
    title: str = ""
    content_text: str = ""
    has_diagram: bool = False
    has_equation: bool = False
    snapshot_url: Optional[str] = None

class GestureEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    gesture_type: GestureType = GestureType.NONE
    intensity: float = 0.0
    duration: float = 0.0
    description: str = ""
    target_region: Optional[str] = None

class VoiceEmphasis(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    pitch_score: float = 0.0
    volume_score: float = 0.0
    rate_score: float = 0.0
    pause_before: float = 0.0
    combined_score: float = 0.0

class ImportanceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    score: float = 0.0
    level: ImportanceLevel = ImportanceLevel.SUPPORTING
    gesture_component: float = 0.0
    voice_component: float = 0.0
    dwell_component: float = 0.0
    keyword_component: float = 0.0
    question_component: float = 0.0
    trigger_text: str = ""

class KeyConcept(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    title: str = ""
    definition: str = ""
    importance_score: float = 0.0
    importance_level: ImportanceLevel = ImportanceLevel.SUPPORTING
    professor_quote: str = ""
    gesture_note: str = ""
    related_concepts: list[str] = []
    slide_number: Optional[int] = None
    sources: list[str] = []

class SummaryUpdate(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    current_topic: str = ""
    bullet_points: list[str] = []
    importance_level: ImportanceLevel = ImportanceLevel.SUPPORTING

class AlertEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    message: str = ""
    alert_type: str = "info"  # info, warning, critical

class Annotation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    lecture_time: str = "00:00:00"
    text: str = ""
    flagged_for_review: bool = False

class ConceptLink(BaseModel):
    source: str
    target: str
    relationship: str = "relates_to"

class ConceptGraph(BaseModel):
    nodes: list[str] = []
    edges: list[ConceptLink] = []


# ── Session Models ───────────────────────────────────────────────
class LectureSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Untitled Lecture"
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: int = 0
    is_active: bool = True
    transcript: list[TranscriptionEvent] = []
    slides: list[SlideEvent] = []
    gestures: list[GestureEvent] = []
    importance_events: list[ImportanceEvent] = []
    key_concepts: list[KeyConcept] = []
    annotations: list[Annotation] = []
    concept_graph: ConceptGraph = Field(default_factory=ConceptGraph)


# ── WebSocket Messages ───────────────────────────────────────────
class WSMessage(BaseModel):
    event_type: EventType
    data: dict
    timestamp: float = Field(default_factory=time.time)


# ── API Requests ─────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    title: str = "Untitled Lecture"
    demo_mode: bool = True

class SearchQuery(BaseModel):
    query: str
    search_type: str = "text"  # text, timestamp, concept
