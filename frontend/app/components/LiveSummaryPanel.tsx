"use client";
import type { SummaryUpdate } from "../types";

interface LiveSummaryPanelProps {
    summary: SummaryUpdate | null;
    isActive: boolean;
}

export default function LiveSummaryPanel({ summary, isActive }: LiveSummaryPanelProps) {
    return (
        <div className="panel summary-panel glass-card">
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">📋</span>
                    Live Summary
                </span>
                {isActive && (
                    <span className="count-badge">LIVE</span>
                )}
            </div>
            <div className="panel-body">
                {summary ? (
                    <>
                        <div className="summary-topic">{summary.current_topic}</div>
                        {summary.bullet_points.map((bp, i) => (
                            <div
                                key={i}
                                className={`summary-bullet ${bp.includes("🔴") || bp.includes("CRITICAL") ? "critical" : ""}`}
                            >
                                {bp}
                            </div>
                        ))}
                    </>
                ) : (
                    <div className="summary-placeholder">
                        <span className="summary-placeholder-icon">🎙️</span>
                        <span>{isActive ? "Waiting for first summary update..." : "Start a session to see live summaries"}</span>
                    </div>
                )}
            </div>
        </div>
    );
}
