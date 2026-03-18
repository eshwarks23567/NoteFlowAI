"""Note Generator — produces structured markdown notes from lecture events."""
from __future__ import annotations
from models.schemas import (
    KeyConcept, TranscriptionEvent, SlideEvent, ImportanceLevel, LectureSession
)
from datetime import datetime


def importance_stars(score: float) -> str:
    if score > 0.85:
        return "⭐⭐⭐"
    elif score > 0.7:
        return "⭐⭐"
    elif score > 0.5:
        return "⭐"
    return ""


def generate_markdown_notes(session: LectureSession) -> str:
    """Generate structured markdown notes from a lecture session."""
    duration = session.duration_seconds
    dur_str = f"{duration // 60} minutes" if duration else "In progress"
    date_str = datetime.fromtimestamp(session.start_time).strftime("%b %d, %Y")

    lines = [
        f"# Lecture: {session.title}",
        f"**Date**: {date_str} | **Duration**: {dur_str}",
        "",
    ]

    # Key concepts section
    critical = [c for c in session.key_concepts if c.importance_level == ImportanceLevel.CRITICAL]
    important = [c for c in session.key_concepts if c.importance_level == ImportanceLevel.IMPORTANT]

    if critical:
        lines.append("## 🔴 Key Concepts (auto-detected)")
        lines.append("")
        for c in critical:
            lines.append(f"### [{c.lecture_time}] {c.title} {importance_stars(c.importance_score)}")
            lines.append(f"- **Definition**: {c.definition}")
            if c.professor_quote:
                lines.append(f'- Professor emphasized: "{c.professor_quote}"')
            if c.gesture_note:
                lines.append(f"- Gestures: {c.gesture_note}")
            if c.related_concepts:
                lines.append(f"- Related: {', '.join(c.related_concepts)}")
            if c.sources:
                lines.append(f"- Sources: {', '.join(c.sources)}")
            lines.append("")

    if important:
        lines.append("## 🟡 Important Points")
        lines.append("")
        for c in important:
            lines.append(f"### [{c.lecture_time}] {c.title} {importance_stars(c.importance_score)}")
            lines.append(f"- **Definition**: {c.definition}")
            if c.professor_quote:
                lines.append(f'- Professor emphasized: "{c.professor_quote}"')
            if c.related_concepts:
                lines.append(f"- Related: {', '.join(c.related_concepts)}")
            lines.append("")

    # Full transcript section
    if session.transcript:
        lines.append("## 📝 Full Transcript")
        lines.append("")
        for t in session.transcript:
            speaker = "👨‍🏫 Professor" if t.speaker.value == "professor" else "🙋 Student"
            emphasis = " **⚡**" if t.is_emphasis_phrase else ""
            lines.append(f"**[{t.lecture_time}]** {speaker}: {t.text}{emphasis}")
            lines.append("")

    return "\n".join(lines)
