"use client";
import { useState, useCallback } from "react";
import { useLectureState } from "./hooks/useLectureState";
import Header from "./components/Header";
import CameraPanel from "./components/CameraPanel";
import LiveSummaryPanel from "./components/LiveSummaryPanel";
import KeyConceptsPanel from "./components/KeyConceptsPanel";
import TranscriptPanel from "./components/TranscriptPanel";
import TimelineBar from "./components/TimelineBar";
import SearchPanel from "./components/SearchPanel";
import AlertsPanel from "./components/AlertsPanel";
import SlideViewer from "./components/SlideViewer";
import ConceptGraph from "./components/ConceptGraph";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const {
    state,
    startSession,
    stopSession,
    exportNotes,
    searchLecture,
    fetchConceptGraph,
  } = useLectureState();

  const [rightTab, setRightTab] = useState<"alerts" | "slide" | "graph">("alerts");

  const handleStartDemo = () => {
    startSession("Machine Learning Basics — Week 3", true);
  };

  const handleStartLive = () => {
    const title = (document.getElementById("lecture-title-input") as HTMLInputElement)?.value?.trim();
    startSession(title || "Live Lecture", false);
  };

  const handleStartYoutube = () => {
    const title = (document.getElementById("lecture-title-input") as HTMLInputElement)?.value?.trim();
    const url = (document.getElementById("youtube-url-input") as HTMLInputElement)?.value?.trim();
    if (!url) {
      alert("Please enter a YouTube URL");
      return;
    }
    startSession(title || "YouTube Lecture", false, url);
  };

  const handleExport = async () => {
    const content = await exportNotes();
    if (content) {
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `lecture-notes-${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleFrameCapture = useCallback(async (frameData: string) => {
    if (!state.sessionId) return;
    try {
      await fetch(`${API_BASE}/api/session/${state.sessionId}/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameData }),
      });
    } catch (e) {
      // Frame processing is best-effort
    }
  }, [state.sessionId]);

  return (
    <div className="app-container">
      <Header
        title={state.title}
        isActive={state.isActive}
        elapsedSeconds={state.elapsedSeconds}
        connectionStatus={state.connectionStatus}
        mode={state.mode}
        notesSavedPath={state.notesSavedPath}
        onStart={handleStartDemo}
        onStartLive={handleStartLive}
        onStop={stopSession}
        onExport={handleExport}
      />

      {/* Start Panel — shown when idle */}
      {!state.isActive && state.mode === "idle" && (
        <div className="start-panel glass-card">
          <div className="start-panel-content">
            <h2 className="start-title">NoteFlow AI</h2>
            <p className="start-subtitle">Secure, real-time capture of lecture audio and slides for structured review.</p>

            <div className="start-input-group">
              <label htmlFor="lecture-title-input" className="start-label">Lecture Title</label>
              <input
                id="lecture-title-input"
                type="text"
                className="start-input"
                placeholder="e.g., Machine Learning — Week 3"
                defaultValue=""
              />
            </div>

            <div className="start-input-group" style={{ marginTop: 16 }}>
              <label htmlFor="youtube-url-input" className="start-label">YouTube Lecture URL</label>
              <input
                id="youtube-url-input"
                type="text"
                className="start-input"
                placeholder="https://www.youtube.com/watch?v=..."
              />
            </div>

            <div className="start-buttons">
              <button className="btn btn-primary start-btn" onClick={handleStartYoutube}>
                Start YouTube Demo
                <span className="btn-sub">Processes video transcript</span>
              </button>
              <button className="btn btn-ghost start-btn" onClick={handleStartLive}>
                Start Live Lecture
                <span className="btn-sub">Uses your mic + camera</span>
              </button>
              <button className="btn btn-ghost start-btn" onClick={handleStartDemo}>
                Run Static Demo
              </button>
            </div>

            <div className="start-features">
              <div className="feature-item">Voice Detection</div>
              <div className="feature-item">Slide OCR</div>
              <div className="feature-item">Smart Notes</div>
              <div className="feature-item">Auto-Save</div>
            </div>
          </div>
        </div>
      )}

      {/* Notes saved notification */}
      {state.notesSavedPath && !state.isActive && (
        <div className="notes-saved-banner glass-card">
          <span>Notes auto-saved!</span>
          <button className="btn btn-primary" onClick={handleExport} style={{ marginLeft: 12 }}>
            Download Notes
          </button>
        </div>
      )}

      {/* Dashboard — shown when active or has data */}
      {(state.isActive || state.transcript.length > 0) && (
        <div className="dashboard-grid">
          {/* ── Left Column: Summary + Key Concepts ── */}
          <LiveSummaryPanel summary={state.summary} isActive={state.isActive} />
          <KeyConceptsPanel concepts={state.keyConcepts} />

          {/* ── Center Column: Camera + Transcript + Timeline ── */}
          <CameraPanel
            isActive={state.isActive}
            onFrameCapture={handleFrameCapture}
          />
          <TranscriptPanel transcript={state.transcript} />
          <TimelineBar
            importanceEvents={state.importanceEvents}
            slides={state.slides}
            elapsedSeconds={state.elapsedSeconds}
            totalDuration={180}
          />

          {/* ── Right Column: Search + Alerts/Slide/Graph ── */}
          <div className="sidebar-right">
            <SearchPanel onSearch={searchLecture} />

            {/* Tab switcher for right panels */}
            <div className="glass-card" style={{ flex: "1 1 auto", display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "12px 18px 0" }}>
                <div className="tab-bar">
                  <button
                    className={`tab-btn ${rightTab === "alerts" ? "active" : ""}`}
                    onClick={() => setRightTab("alerts")}
                  >
                    Alerts {state.alerts.length > 0 && `(${state.alerts.length})`}
                  </button>
                  <button
                    className={`tab-btn ${rightTab === "slide" ? "active" : ""}`}
                    onClick={() => setRightTab("slide")}
                  >
                    Slide
                  </button>
                  <button
                    className={`tab-btn ${rightTab === "graph" ? "active" : ""}`}
                    onClick={() => setRightTab("graph")}
                  >
                    Graph
                  </button>
                </div>
              </div>

              <div style={{ flex: 1, overflow: "auto" }}>
                {rightTab === "alerts" && <AlertsPanel alerts={state.alerts} />}
                {rightTab === "slide" && <SlideViewer currentSlide={state.currentSlide} />}
                {rightTab === "graph" && (
                  <ConceptGraph
                    graph={state.conceptGraph}
                    onFetch={fetchConceptGraph}
                    isActive={state.isActive}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
