"use client";
import { formatLectureTime } from "../utils/formatters";

interface HeaderProps {
    title: string;
    isActive: boolean;
    elapsedSeconds: number;
    connectionStatus: string;
    mode: "idle" | "demo" | "live" | "youtube";
    notesSavedPath: string | null;
    onStart: () => void;
    onStartLive: () => void;
    onStop: () => void;
    onExport: () => void;
}

export default function Header({
    title, isActive, elapsedSeconds, connectionStatus, mode, notesSavedPath,
    onStart, onStartLive, onStop, onExport,
}: HeaderProps) {
    return (
        <header className="app-header">
            <div className="header-left">
                <div className="header-logo">
                    <span className="header-logo-icon">N</span>
                    NoteFlow AI
                </div>
                <span className="header-title">{title}</span>
                {isActive && mode !== "idle" && (
                    <span className={`mode-badge ${mode}`}>
                        {mode === "live" ? "LIVE" : mode === "youtube" ? "YOUTUBE" : "DEMO"}
                    </span>
                )}
            </div>

            <div className="header-right">
                {isActive && (
                    <span className="timer">{formatLectureTime(elapsedSeconds)}</span>
                )}

                <span className={`status-badge ${isActive ? "live" : "offline"}`}>
                    {isActive && <span className="pulse-dot" />}
                    {isActive ? "LIVE" : connectionStatus === "connected" ? "READY" : "OFFLINE"}
                </span>

                {isActive && (
                    <button className="btn btn-danger" onClick={onStop}>
                        ■ Stop
                    </button>
                )}

                <button className="btn btn-ghost" onClick={onExport} title="Export Notes">
                    Export
                </button>
            </div>
        </header>
    );
}
