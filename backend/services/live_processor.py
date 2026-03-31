"""Live Processor — handles real-time transcript + frame processing for live mode."""
from __future__ import annotations
import base64, io, re, time, hashlib, asyncio
from typing import Callable, Awaitable, Optional
from collections import Counter

from models.schemas import (
    TranscriptionEvent, SlideEvent, GestureEvent, ImportanceEvent,
    KeyConcept, SummaryUpdate, AlertEvent,
    ImportanceLevel, SpeakerRole, GestureType, EventType, WSMessage,
    LectureSession,
)
from services.fusion_engine import fuse_scores, compute_keyword_score, classify_importance


# ── OCR helpers ─────────────────────────────────────────────────
_tesseract_available = False
_cv2_available = False

try:
    import pytesseract
    _tesseract_available = True
except ImportError:
    pass

try:
    import cv2
    import numpy as np
    _cv2_available = True
except ImportError:
    pass


# ── Concept extraction keywords ─────────────────────────────────
CONCEPT_INDICATORS = [
    "is defined as", "refers to", "means that", "is a", "is the",
    "are called", "known as", "consists of", "involves",
    "the key", "important", "crucial", "essential", "fundamental",
    "remember", "pay attention", "don't forget", "note that",
    "for example", "such as", "in other words",
]

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "for", "yet", "so", "in", "on", "at",
    "to", "of", "by", "with", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "if", "then", "else", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "than", "too", "very", "just",
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "its", "his", "her", "our", "their", "us", "them",
    "going", "let", "let's", "like", "also", "now", "right", "here",
    "okay", "ok", "so", "well", "actually", "basically", "really",
    "um", "uh", "yeah", "yes", "no", "think", "know", "want", "get",
    "got", "make", "take", "see", "come", "go", "say", "said",
}


class LiveProcessor:
    """Processes speech transcripts and camera frames in real-time."""

    def __init__(self, send_callback: Optional[Callable[[WSMessage], Awaitable[None]]] = None):
        self.send = send_callback
        self.start_time = time.time()
        self._last_ocr_hash = ""
        self._last_slide_number = 0
        self._word_counter: Counter = Counter()
        self._concept_titles_seen: set[str] = set()
        self._transcript_buffer: list[str] = []
        self._summary_counter = 0

    def _lecture_time(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── Transcript processing ───────────────────────────────────
    async def process_transcript(
        self, text: str, session: LectureSession
    ) -> dict:
        """Process transcribed text from browser Speech API."""
        if not text or not text.strip():
            return {"status": "empty"}

        text = text.strip()
        lt = self._lecture_time()

        # Detect emphasis phrases
        is_emphasis = compute_keyword_score(text) > 0
        keywords = self._extract_keywords(text)

        # Create transcription event
        t_event = TranscriptionEvent(
            text=text,
            speaker=SpeakerRole.PROFESSOR,
            confidence=0.90,
            is_emphasis_phrase=is_emphasis,
            keywords=keywords,
            lecture_time=lt,
        )
        session.transcript.append(t_event)

        # Broadcast transcription
        if self.send:
            await self.send(WSMessage(
                event_type=EventType.TRANSCRIPTION,
                data=t_event.model_dump(),
            ))

        # Update word frequency for concept detection
        for w in keywords:
            self._word_counter[w.lower()] += 1

        self._transcript_buffer.append(text)

        # Compute importance score
        kw_score = compute_keyword_score(text)
        voice_emphasis = 0.6 if is_emphasis else 0.2

        if is_emphasis or kw_score > 0.3:
            imp = fuse_scores(
                gesture_intensity=0.0,
                voice_emphasis=voice_emphasis,
                slide_dwell_time=0.3,
                keyword_score=kw_score,
                question_score=0.0,
                lecture_time=lt,
                trigger_text=text[:100],
            )
            session.importance_events.append(imp)
            if self.send:
                await self.send(WSMessage(
                    event_type=EventType.IMPORTANCE,
                    data=imp.model_dump(),
                ))

            # Alert for critical moments
            if imp.level == ImportanceLevel.CRITICAL and self.send:
                alert = AlertEvent(
                    message=f"Important: {text[:60]}...",
                    alert_type="critical",
                    lecture_time=lt,
                )
                await self.send(WSMessage(
                    event_type=EventType.ALERT,
                    data=alert.model_dump(),
                ))

        # Try to extract concepts
        concept = self._try_extract_concept(text, keywords, kw_score, lt)
        if concept:
            session.key_concepts.append(concept)
            if self.send:
                await self.send(WSMessage(
                    event_type=EventType.KEY_CONCEPT,
                    data=concept.model_dump(),
                ))

        # Periodic summary
        if len(self._transcript_buffer) >= 5:
            await self._send_summary(session)

        return {"status": "processed", "keywords": keywords, "is_emphasis": is_emphasis}

    # ── Frame / OCR processing ──────────────────────────────────
    async def process_frame(
        self, frame_base64: str, session: LectureSession
    ) -> dict:
        """Process a camera frame — extract text via OCR."""
        if not _tesseract_available or not _cv2_available:
            return {"status": "ocr_unavailable", "message": "Tesseract or OpenCV not installed"}

        try:
            # Strip data URL prefix if present
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",", 1)[1]

            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"status": "decode_error"}

            # Convert to grayscale and enhance for OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adaptive threshold for better OCR on projected slides
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Run Tesseract
            ocr_text = pytesseract.image_to_string(gray, config="--psm 6")
            ocr_text = ocr_text.strip()

            if not ocr_text or len(ocr_text) < 10:
                return {"status": "no_text_detected"}

            # Check if content changed significantly (new slide)
            text_hash = hashlib.md5(ocr_text.encode()).hexdigest()[:12]
            if text_hash == self._last_ocr_hash:
                return {"status": "unchanged"}

            self._last_ocr_hash = text_hash
            self._last_slide_number += 1

            lt = self._lecture_time()

            # Extract a title from the first line
            lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
            title = lines[0][:80] if lines else "Slide Content"
            content = "\n".join(lines[1:]) if len(lines) > 1 else ocr_text

            # Create slide event
            slide_event = SlideEvent(
                slide_number=self._last_slide_number,
                title=title,
                content_text=content,
                has_diagram=False,
                has_equation=bool(re.search(r'[=∑∫±×÷²³]', ocr_text)),
                lecture_time=lt,
            )
            session.slides.append(slide_event)

            if self.send:
                await self.send(WSMessage(
                    event_type=EventType.SLIDE_CHANGE,
                    data=slide_event.model_dump(),
                ))

            # Extract keywords from OCR text for importance scoring
            ocr_keywords = self._extract_keywords(ocr_text)
            kw_score = compute_keyword_score(ocr_text)

            if kw_score > 0 or len(ocr_keywords) >= 3:
                imp = fuse_scores(
                    gesture_intensity=0.0,
                    voice_emphasis=0.0,
                    slide_dwell_time=0.5,
                    keyword_score=kw_score,
                    question_score=0.0,
                    lecture_time=lt,
                    trigger_text=f"[Slide {self._last_slide_number}] {title[:60]}",
                )
                session.importance_events.append(imp)
                if self.send:
                    await self.send(WSMessage(
                        event_type=EventType.IMPORTANCE,
                        data=imp.model_dump(),
                    ))

            return {
                "status": "processed",
                "slide_number": self._last_slide_number,
                "title": title,
                "text_length": len(ocr_text),
                "keywords": ocr_keywords,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Concept extraction ──────────────────────────────────────
    def _try_extract_concept(
        self, text: str, keywords: list[str], kw_score: float, lecture_time: str
    ) -> Optional[KeyConcept]:
        """Try to extract a key concept from the text using keyword analysis."""
        if not keywords:
            return None

        text_lower = text.lower()

        # Check if the text contains concept-defining patterns
        has_definition = any(ind in text_lower for ind in CONCEPT_INDICATORS)
        has_emphasis = kw_score > 0

        if not has_definition and not has_emphasis:
            return None

        # Pick the most significant keyword as the concept title
        # Prefer multi-word terms, or most frequent single words
        best_title = None
        best_score = 0
        for kw in keywords:
            # Score based on: length, frequency, not seen before
            freq = self._word_counter.get(kw.lower(), 1)
            length_bonus = min(3, len(kw.split())) * 0.3
            novelty = 0.5 if kw.lower() not in self._concept_titles_seen else 0.0
            score = freq * 0.3 + length_bonus + novelty + (0.3 if has_definition else 0)
            if score > best_score:
                best_score = score
                best_title = kw

        if not best_title or best_title.lower() in self._concept_titles_seen:
            return None

        # Build definition from the sentence
        importance = min(0.95, 0.4 + kw_score * 0.3 + (0.2 if has_definition else 0) + best_score * 0.05)

        self._concept_titles_seen.add(best_title.lower())

        related = [k for k in keywords if k.lower() != best_title.lower()][:4]

        concept = KeyConcept(
            title=best_title.title(),
            definition=text[:200],
            importance_score=round(importance, 3),
            importance_level=classify_importance(importance),
            professor_quote=text[:120] if has_emphasis else "",
            gesture_note="",
            related_concepts=related,
            slide_number=self._last_slide_number if self._last_slide_number > 0 else None,
            sources=["voice"] + (["slide"] if self._last_slide_number > 0 else []),
            lecture_time=lecture_time,
        )
        return concept

    # ── Keyword extraction ──────────────────────────────────────
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text using simple NLP."""
        # Tokenize
        words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text)
        # Filter stop words
        filtered = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]

        # Find bigrams (consecutive non-stopword pairs)
        bigrams = []
        for i in range(len(filtered) - 1):
            bigram = f"{filtered[i]} {filtered[i+1]}"
            if len(bigram) > 6:
                bigrams.append(bigram)

        # Combine, deduplicate, return top keywords
        all_terms = bigrams[:3] + filtered[:6]
        seen = set()
        result = []
        for term in all_terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                result.append(term)

        return result[:8]

    # ── Summary generation ──────────────────────────────────────
    async def _send_summary(self, session: LectureSession):
        """Generate and send a summary update from buffered transcripts."""
        if not self._transcript_buffer:
            return

        self._summary_counter += 1
        buffer_text = " ".join(self._transcript_buffer)
        self._transcript_buffer.clear()

        # Create bullet points from the most important sentences
        sentences = re.split(r'[.!?]+', buffer_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        # Pick top sentences based on keyword presence
        scored = []
        for s in sentences:
            score = compute_keyword_score(s)
            kw_count = len(self._extract_keywords(s))
            scored.append((score + kw_count * 0.1, s))
        scored.sort(reverse=True)

        bullets = [s[:120] for _, s in scored[:4]]
        if not bullets:
            bullets = [s[:120] for s in sentences[:3]]

        # Determine current topic from most recent concept
        current_topic = "Lecture in Progress"
        if session.key_concepts:
            current_topic = session.key_concepts[-1].title

        level = ImportanceLevel.IMPORTANT
        if any(compute_keyword_score(b) > 0.3 for b in bullets):
            level = ImportanceLevel.CRITICAL

        su = SummaryUpdate(
            current_topic=current_topic,
            bullet_points=bullets,
            importance_level=level,
        )

        if self.send:
            await self.send(WSMessage(
                event_type=EventType.SUMMARY_UPDATE,
                data=su.model_dump(),
            ))
