"use client";
import type { ImportanceEvent, SlideEvent } from "../types";
import { formatLectureTime } from "../utils/formatters";

interface TimelineBarProps {
    importanceEvents: ImportanceEvent[];
    slides: SlideEvent[];
    elapsedSeconds: number;
    totalDuration?: number;
}

export default function TimelineBar({
    importanceEvents, slides, elapsedSeconds, totalDuration = 3000,
}: TimelineBarProps) {
    const progress = Math.min(100, (elapsedSeconds / totalDuration) * 100);

    return (
        <div className="panel timeline-panel glass-card">
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">TL</span>
                    Timeline
                </span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "var(--text-muted)" }}>
                    {formatLectureTime(elapsedSeconds)}
                </span>
            </div>
            <div style={{ padding: "12px 18px" }}>
                <div className="timeline-bar">
                    <div className="timeline-progress" style={{ width: `${progress}%` }} />
                    <div className="timeline-markers">
                        {/* Slide change markers */}
                        {slides.map((slide, i) => {
                            const pos = ((slide.timestamp - (Date.now() / 1000 - elapsedSeconds)) / totalDuration) * 100;
                            if (pos < 0 || pos > 100) return null;
                            return (
                                <div
                                    key={`slide-${i}`}
                                    className="timeline-marker slide-change"
                                    style={{ left: `${pos}%` }}
                                    title={`Slide ${slide.slide_number}: ${slide.title}`}
                                />
                            );
                        })}
                        {/* Importance markers */}
                        {importanceEvents.map((evt, i) => {
                            const pos = ((evt.timestamp - (Date.now() / 1000 - elapsedSeconds)) / totalDuration) * 100;
                            if (pos < 0 || pos > 100) return null;
                            return (
                                <div
                                    key={`imp-${i}`}
                                    className={`timeline-marker ${evt.level}`}
                                    style={{
                                        left: `${pos}%`,
                                        height: `${Math.max(20, evt.score * 100)}%`,
                                    }}
                                    title={`${evt.level}: ${evt.trigger_text}`}
                                />
                            );
                        })}
                    </div>
                </div>
                <div className="timeline-labels">
                    <span>00:00:00</span>
                    <span>{formatLectureTime(Math.floor(totalDuration / 4))}</span>
                    <span>{formatLectureTime(Math.floor(totalDuration / 2))}</span>
                    <span>{formatLectureTime(Math.floor((totalDuration * 3) / 4))}</span>
                    <span>{formatLectureTime(totalDuration)}</span>
                </div>
            </div>
        </div>
    );
}
