"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import type {
    LectureState, TranscriptionEvent, SlideEvent, GestureEvent,
    ImportanceEvent, KeyConcept, AlertEvent, SummaryUpdate, WSMessage
} from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live";

const initialState: LectureState = {
    sessionId: null,
    isActive: false,
    title: "Live Lecture Note-Taker",
    startTime: 0,
    elapsedSeconds: 0,
    transcript: [],
    slides: [],
    currentSlide: null,
    gestures: [],
    importanceEvents: [],
    keyConcepts: [],
    alerts: [],
    annotations: [],
    summary: null,
    conceptGraph: null,
    connectionStatus: "disconnected",
};

export function useLectureState() {
    const [state, setState] = useState<LectureState>(initialState);
    const wsRef = useRef<WebSocket | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    /* ── WebSocket connection ── */
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;
        setState(s => ({ ...s, connectionStatus: "connecting" }));
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
            setState(s => ({ ...s, connectionStatus: "connected" }));
        };

        ws.onclose = () => {
            setState(s => ({ ...s, connectionStatus: "disconnected" }));
            // Reconnect after 3 seconds
            setTimeout(() => {
                if (wsRef.current?.readyState !== WebSocket.OPEN) connect();
            }, 3000);
        };

        ws.onerror = () => {
            ws.close();
        };

        ws.onmessage = (event) => {
            try {
                const msg: WSMessage = JSON.parse(event.data);
                handleEvent(msg);
            } catch (e) {
                console.error("Failed to parse WS message:", e);
            }
        };
    }, []);

    const handleEvent = (msg: WSMessage) => {
        const { event_type, data } = msg;

        setState(prev => {
            switch (event_type) {
                case "transcription":
                    return {
                        ...prev,
                        transcript: [...prev.transcript, data as unknown as TranscriptionEvent],
                    };

                case "slide_change":
                    const slide = data as unknown as SlideEvent;
                    return {
                        ...prev,
                        slides: [...prev.slides, slide],
                        currentSlide: slide,
                    };

                case "gesture":
                    return {
                        ...prev,
                        gestures: [...prev.gestures, data as unknown as GestureEvent],
                    };

                case "importance":
                    return {
                        ...prev,
                        importanceEvents: [...prev.importanceEvents, data as unknown as ImportanceEvent],
                    };

                case "key_concept":
                    return {
                        ...prev,
                        keyConcepts: [...prev.keyConcepts, data as unknown as KeyConcept],
                    };

                case "alert":
                    const alert = data as unknown as AlertEvent;
                    return {
                        ...prev,
                        alerts: [alert, ...prev.alerts].slice(0, 20),
                    };

                case "summary_update":
                    return {
                        ...prev,
                        summary: data as unknown as SummaryUpdate,
                    };

                case "session_status":
                    if ((data as any).status === "started") {
                        return {
                            ...prev,
                            isActive: true,
                            title: (data as any).title || prev.title,
                            startTime: Date.now() / 1000,
                        };
                    }
                    if ((data as any).status === "stopped") {
                        return { ...prev, isActive: false };
                    }
                    return prev;

                default:
                    return prev;
            }
        });
    };

    /* ── Session management ── */
    const startSession = async (title: string = "Untitled Lecture", demoMode: boolean = true) => {
        try {
            const res = await fetch(`${API_BASE}/api/session/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title, demo_mode: demoMode }),
            });
            const data = await res.json();
            setState(s => ({
                ...s,
                sessionId: data.session_id,
                title: data.title,
                isActive: true,
                startTime: Date.now() / 1000,
                // Reset state
                transcript: [],
                slides: [],
                currentSlide: null,
                gestures: [],
                importanceEvents: [],
                keyConcepts: [],
                alerts: [],
                annotations: [],
                summary: null,
                conceptGraph: null,
            }));
            connect();
        } catch (e) {
            console.error("Failed to start session:", e);
        }
    };

    const stopSession = async () => {
        if (!state.sessionId) return;
        try {
            await fetch(`${API_BASE}/api/session/${state.sessionId}/stop`, { method: "POST" });
            setState(s => ({ ...s, isActive: false }));
        } catch (e) {
            console.error("Failed to stop session:", e);
        }
    };

    const exportNotes = async (): Promise<string | null> => {
        if (!state.sessionId) return null;
        try {
            const res = await fetch(`${API_BASE}/api/session/${state.sessionId}/notes`);
            const data = await res.json();
            return data.markdown;
        } catch (e) {
            console.error("Failed to export notes:", e);
            return null;
        }
    };

    const searchLecture = async (query: string) => {
        if (!state.sessionId) return [];
        try {
            const res = await fetch(`${API_BASE}/api/session/${state.sessionId}/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            });
            const data = await res.json();
            return data.results;
        } catch (e) {
            console.error("Failed to search:", e);
            return [];
        }
    };

    const fetchConceptGraph = async () => {
        if (!state.sessionId) return;
        try {
            const res = await fetch(`${API_BASE}/api/session/${state.sessionId}/concepts`);
            const data = await res.json();
            setState(s => ({ ...s, conceptGraph: data.graph }));
        } catch (e) {
            console.error("Failed to fetch concept graph:", e);
        }
    };

    /* ── Elapsed timer ── */
    useEffect(() => {
        if (state.isActive) {
            timerRef.current = setInterval(() => {
                setState(s => ({
                    ...s,
                    elapsedSeconds: Math.floor(Date.now() / 1000 - s.startTime),
                }));
            }, 1000);
        } else if (timerRef.current) {
            clearInterval(timerRef.current);
        }
        return () => { if (timerRef.current) clearInterval(timerRef.current); };
    }, [state.isActive]);

    /* ── Web Speech API for Local Transcription ── */
    const recognitionRef = useRef<any>(null);

    useEffect(() => {
        if (typeof window !== "undefined" && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
            const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = false;

            recognitionRef.current.onresult = (event: any) => {
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        const text = event.results[i][0].transcript;
                        const tEvent = {
                            id: `local-${Date.now()}-${i}`,
                            session_id: "local",
                            speaker: "professor",
                            text: text.trim(),
                            start_time: 0,
                            end_time: 0,
                            lecture_time: new Date().toLocaleTimeString('en-US', { hour12: false }),
                            is_emphasis_phrase: false,
                            keywords: []
                        };
                        setState(prev => ({
                            ...prev,
                            // Prepend local transcripts to simulated ones to mix them, or just append
                            transcript: [...prev.transcript, tEvent as any],
                        }));
                    }
                }
            };
            recognitionRef.current.onerror = (e: any) => console.error("Speech Recognition Error:", e);
        }
    }, []);

    useEffect(() => {
        if (state.isActive && recognitionRef.current) {
            try { recognitionRef.current.start(); } catch (e) { }
        } else if (!state.isActive && recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch (e) { }
        }
    }, [state.isActive]);

    /* ── Cleanup ── */
    useEffect(() => {
        return () => {
            wsRef.current?.close();
            if (timerRef.current) clearInterval(timerRef.current);
            if (recognitionRef.current) {
                try { recognitionRef.current.stop(); } catch (e) { }
            }
        };
    }, []);

    return {
        state,
        startSession,
        stopSession,
        exportNotes,
        searchLecture,
        fetchConceptGraph,
    };
}
