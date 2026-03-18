"use client";
import { useEffect, useRef } from "react";
import type { TranscriptionEvent } from "../types";
import { speakerIcon } from "../utils/formatters";

interface TranscriptPanelProps {
    transcript: TranscriptionEvent[];
}

export default function TranscriptPanel({ transcript }: TranscriptPanelProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [transcript.length]);

    return (
        <div className="panel transcript-panel glass-card">
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">📝</span>
                    Live Transcript
                </span>
                {transcript.length > 0 && (
                    <span className="count-badge">{transcript.length}</span>
                )}
            </div>
            <div className="panel-body">
                {transcript.length === 0 ? (
                    <div className="empty-state">
                        <span className="empty-icon">🎙️</span>
                        <span className="empty-text">Live transcription will appear here as the professor speaks</span>
                    </div>
                ) : (
                    transcript.map((entry) => (
                        <div key={entry.id} className="transcript-entry">
                            <span className="transcript-speaker-icon">
                                {speakerIcon(entry.speaker)}
                            </span>
                            <div className="transcript-content">
                                <div className="transcript-meta">
                                    <span className={`transcript-speaker ${entry.speaker}`}>
                                        {entry.speaker}
                                    </span>
                                    <span className="transcript-time">{entry.lecture_time}</span>
                                    {entry.is_emphasis_phrase && (
                                        <span className="count-badge critical">⚡</span>
                                    )}
                                </div>
                                <div className={`transcript-text ${entry.is_emphasis_phrase ? "emphasis" : ""}`}>
                                    {entry.text}
                                </div>
                                {entry.keywords.length > 0 && (
                                    <div className="transcript-keywords">
                                        {entry.keywords.map((kw, i) => (
                                            <span key={i} className="keyword-tag">{kw}</span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
                <div ref={bottomRef} />
            </div>
        </div>
    );
}
