"""FastAPI backend for the Live Lecture Note-Taker."""
from __future__ import annotations
import asyncio, json, time, sys, os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from models.schemas import (
    LectureSession, WSMessage, EventType, TranscriptionEvent, SlideEvent,
    GestureEvent, ImportanceEvent, KeyConcept, Annotation, SearchQuery,
    ConceptGraph, ConceptLink, StartSessionRequest
)
from services.demo_simulator import (
    DemoSimulator, CONCEPT_GRAPH_NODES, CONCEPT_GRAPH_EDGES
)
from services.note_generator import generate_markdown_notes


# ── State ────────────────────────────────────────────────────────
active_sessions: dict[str, LectureSession] = {}
connected_clients: list[WebSocket] = []
active_simulators: dict[str, DemoSimulator] = {}


# ── App ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🎓 Live Lecture Note-Taker Backend — Ready")
    print("   Demo mode available at POST /api/session/start")
    yield
    # Cleanup
    for sim in active_simulators.values():
        await sim.stop()

app = FastAPI(
    title="Live Lecture Note-Taker",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────
async def broadcast(message: WSMessage):
    """Send a message to all connected WebSocket clients."""
    data = message.model_dump()
    data["data"] = message.data  # preserve original dict
    payload = json.dumps(data, default=str)
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


def _store_event(session_id: str, event_type: EventType, data: dict):
    """Store an event in the session history."""
    session = active_sessions.get(session_id)
    if not session:
        return
    if event_type == EventType.TRANSCRIPTION:
        session.transcript.append(TranscriptionEvent(**data))
    elif event_type == EventType.SLIDE_CHANGE:
        session.slides.append(SlideEvent(**data))
    elif event_type == EventType.GESTURE:
        session.gestures.append(GestureEvent(**data))
    elif event_type == EventType.IMPORTANCE:
        session.importance_events.append(ImportanceEvent(**data))
    elif event_type == EventType.KEY_CONCEPT:
        session.key_concepts.append(KeyConcept(**data))


# ── REST Endpoints ───────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(active_sessions)}


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    session = LectureSession(title=req.title)
    active_sessions[session.id] = session

    if req.demo_mode:
        async def store_and_broadcast(msg: WSMessage):
            _store_event(session.id, msg.event_type, msg.data)
            await broadcast(msg)

        sim = DemoSimulator(send_callback=store_and_broadcast)
        active_simulators[session.id] = sim
        asyncio.create_task(sim.start())

    return {"session_id": session.id, "title": session.title, "demo_mode": req.demo_mode}


@app.post("/api/session/{session_id}/stop")
async def stop_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.is_active = False
    session.end_time = time.time()
    session.duration_seconds = int(session.end_time - session.start_time)

    sim = active_simulators.pop(session_id, None)
    if sim:
        await sim.stop()

    return {"session_id": session_id, "duration": session.duration_seconds}


@app.get("/api/session/{session_id}/notes")
async def get_notes(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    markdown = generate_markdown_notes(session)
    return {"session_id": session_id, "markdown": markdown}


@app.get("/api/session/{session_id}/concepts")
async def get_concepts(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    # Build concept graph
    graph = ConceptGraph(
        nodes=CONCEPT_GRAPH_NODES,
        edges=[ConceptLink(source=s, target=t, relationship=r) for s, t, r in CONCEPT_GRAPH_EDGES]
    )
    return {
        "concepts": [c.model_dump() for c in session.key_concepts],
        "graph": graph.model_dump()
    }


@app.post("/api/session/{session_id}/annotate")
async def add_annotation(session_id: str, annotation: Annotation):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.annotations.append(annotation)
    return {"status": "ok", "annotation_id": annotation.id}


@app.post("/api/session/{session_id}/search")
async def search_session(session_id: str, query: SearchQuery):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    q = query.query.lower()
    results = []
    for t in session.transcript:
        if q in t.text.lower():
            results.append({
                "type": "transcript",
                "lecture_time": t.lecture_time,
                "text": t.text,
                "speaker": t.speaker.value,
            })
    for c in session.key_concepts:
        if q in c.title.lower() or q in c.definition.lower():
            results.append({
                "type": "concept",
                "lecture_time": c.lecture_time,
                "title": c.title,
                "definition": c.definition,
            })
    return {"query": query.query, "results": results, "count": len(results)}


# ── WebSocket ────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print(f"🔌 Client connected ({len(connected_clients)} total)")
    try:
        while True:
            # Keep connection alive; handle client messages
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle annotation from client
            if msg.get("type") == "annotation":
                for session in active_sessions.values():
                    if session.is_active:
                        session.annotations.append(Annotation(
                            text=msg.get("text", ""),
                            lecture_time=msg.get("lecture_time", "00:00:00"),
                            flagged_for_review=msg.get("flagged", False),
                        ))
    except WebSocketDisconnect:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)
        print(f"🔌 Client disconnected ({len(connected_clients)} total)")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
