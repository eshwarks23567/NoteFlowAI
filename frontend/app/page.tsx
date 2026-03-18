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

  const handleStart = () => {
    startSession("Machine Learning Basics — Week 3", true);
  };

  const handleExport = async () => {
    const md = await exportNotes();
    if (md) {
      const blob = new Blob([md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `lecture-notes-${new Date().toISOString().slice(0, 10)}.md`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleFrameCapture = useCallback(async (frameData: string) => {
    // Send captured frame to the backend for OCR / gesture processing
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
        onStart={handleStart}
        onStop={stopSession}
        onExport={handleExport}
      />

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
                  🔔 Alerts {state.alerts.length > 0 && `(${state.alerts.length})`}
                </button>
                <button
                  className={`tab-btn ${rightTab === "slide" ? "active" : ""}`}
                  onClick={() => setRightTab("slide")}
                >
                  🖥️ Slide
                </button>
                <button
                  className={`tab-btn ${rightTab === "graph" ? "active" : ""}`}
                  onClick={() => setRightTab("graph")}
                >
                  🕸️ Graph
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
    </div>
  );
}
