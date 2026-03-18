"use client";
import { formatLectureTime } from "../utils/formatters";

interface HeaderProps {
    title: string;
    isActive: boolean;
    elapsedSeconds: number;
    connectionStatus: string;
    onStart: () => void;
    onStop: () => void;
    onExport: () => void;
}

export default function Header({
    title, isActive, elapsedSeconds, connectionStatus, onStart, onStop, onExport,
}: HeaderProps) {
    return (
        <header className="app-header">
            <div className="header-left">
                <div className="header-logo">
                    <span className="header-logo-icon">🎓</span>
                    NoteFlow AI
                </div>
                <span className="header-title">{title}</span>
            </div>

            <div className="header-right">
                {isActive && (
                    <span className="timer">{formatLectureTime(elapsedSeconds)}</span>
                )}

                <span className={`status-badge ${isActive ? "live" : "offline"}`}>
                    {isActive && <span className="pulse-dot" />}
                    {isActive ? "LIVE" : connectionStatus === "connected" ? "READY" : "OFFLINE"}
                </span>

                {!isActive ? (
                    <button className="btn btn-primary" onClick={onStart}>
                        ▶ Start Demo
                    </button>
                ) : (
                    <button className="btn btn-danger" onClick={onStop}>
                        ■ Stop
                    </button>
                )}

                <button className="btn btn-ghost" onClick={onExport} title="Export Notes">
                    📄 Export
                </button>
            </div>
        </header>
    );
}
