/* ── TypeScript types mirroring backend Pydantic models ── */

export type ImportanceLevel = "critical" | "important" | "supporting";
export type SpeakerRole = "professor" | "student" | "unknown";
export type GestureType = "pointing" | "sweeping" | "counting" | "emphasis" | "hands_raised" | "leaning_forward" | "none";
export type EventType = "transcription" | "slide_change" | "gesture" | "importance" | "key_concept" | "alert" | "session_status" | "summary_update";

export interface TranscriptionEvent {
  id: string;
  timestamp: number;
  lecture_time: string;
  text: string;
  speaker: SpeakerRole;
  confidence: number;
  is_emphasis_phrase: boolean;
  keywords: string[];
}

export interface SlideEvent {
  id: string;
  timestamp: number;
  lecture_time: string;
  slide_number: number;
  title: string;
  content_text: string;
  has_diagram: boolean;
  has_equation: boolean;
  snapshot_url?: string;
}

export interface GestureEvent {
  id: string;
  timestamp: number;
  lecture_time: string;
  gesture_type: GestureType;
  intensity: number;
  duration: number;
  description: string;
  target_region?: string;
}

export interface ImportanceEvent {
  id: string;
  timestamp: number;
  lecture_time: string;
  score: number;
  level: ImportanceLevel;
  gesture_component: number;
  voice_component: number;
  dwell_component: number;
  keyword_component: number;
  question_component: number;
  trigger_text: string;
}

export interface KeyConcept {
  id: string;
  timestamp: number;
  lecture_time: string;
  title: string;
  definition: string;
  importance_score: number;
  importance_level: ImportanceLevel;
  professor_quote: string;
  gesture_note: string;
  related_concepts: string[];
  slide_number?: number;
  sources: string[];
}

export interface SummaryUpdate {
  timestamp: number;
  current_topic: string;
  bullet_points: string[];
  importance_level: ImportanceLevel;
}

export interface AlertEvent {
  id: string;
  timestamp: number;
  lecture_time: string;
  message: string;
  alert_type: "info" | "warning" | "critical";
}

export interface Annotation {
  id: string;
  timestamp: number;
  lecture_time: string;
  text: string;
  flagged_for_review: boolean;
}

export interface ConceptLink {
  source: string;
  target: string;
  relationship: string;
}

export interface ConceptGraph {
  nodes: string[];
  edges: ConceptLink[];
}

export interface WSMessage {
  event_type: EventType;
  data: Record<string, unknown>;
  timestamp: number;
}

export interface LectureState {
  sessionId: string | null;
  isActive: boolean;
  title: string;
  startTime: number;
  elapsedSeconds: number;
  transcript: TranscriptionEvent[];
  slides: SlideEvent[];
  currentSlide: SlideEvent | null;
  gestures: GestureEvent[];
  importanceEvents: ImportanceEvent[];
  keyConcepts: KeyConcept[];
  alerts: AlertEvent[];
  annotations: Annotation[];
  summary: SummaryUpdate | null;
  conceptGraph: ConceptGraph | null;
  connectionStatus: "disconnected" | "connecting" | "connected";
  mode: "idle" | "demo" | "live";
  notesSavedPath: string | null;
}
