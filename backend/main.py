"""FastAPI backend for the Live Lecture Note-Taker."""
from __future__ import annotations
import asyncio, json, time, sys, os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from services.note_generator import generate_markdown_notes, save_notes_to_file
from services.live_processor import LiveProcessor
from services.youtube_processor import YoutubeProcessor
from services.llm_service import llm_service


# ── State ────────────────────────────────────────────────────────
active_sessions: dict[str, LectureSession] = {}
connected_clients: list[WebSocket] = []
active_simulators: dict[str, DemoSimulator] = {}
active_processors: dict[str, LiveProcessor] = {}
active_youtube_processors: dict[str, YoutubeProcessor] = {}


# ── App ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create notes output directory
    os.makedirs(os.path.join(os.path.dirname(__file__), "notes"), exist_ok=True)
    print("Live Lecture Note-Taker Backend -- Ready")
    print("   Demo mode: POST /api/session/start  {demo_mode: true}")
    print("   Live mode: POST /api/session/start  {demo_mode: false}")
    yield
    # Cleanup
    for sim in active_simulators.values():
        await sim.stop()

app = FastAPI(
    title="Live Lecture Note-Taker",
    version="2.0.0",
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

    if req.youtube_url:
        # YouTube mode
        async def store_and_broadcast(msg: WSMessage):
            _store_event(session.id, msg.event_type, msg.data)
            await broadcast(msg)
            
        yt_processor = YoutubeProcessor(send_callback=store_and_broadcast)
        active_youtube_processors[session.id] = yt_processor
        asyncio.create_task(yt_processor.process_video(req.youtube_url, session))
    elif req.demo_mode:
        # Demo mode — use the simulator with hardcoded data
        async def store_and_broadcast(msg: WSMessage):
            _store_event(session.id, msg.event_type, msg.data)
            await broadcast(msg)

        sim = DemoSimulator(send_callback=store_and_broadcast)
        active_simulators[session.id] = sim
        asyncio.create_task(sim.start())
    else:
        # Live mode — create a processor for real-time data
        processor = LiveProcessor(send_callback=broadcast)
        active_processors[session.id] = processor

    return {
        "session_id": session.id,
        "title": session.title,
        "demo_mode": req.demo_mode,
        "youtube_url": req.youtube_url,
        "mode": "youtube" if req.youtube_url else "demo" if req.demo_mode else "live",
    }


@app.post("/api/session/{session_id}/stop")
async def stop_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.is_active = False
    session.end_time = time.time()
    session.duration_seconds = int(session.end_time - session.start_time)

    # Stop YouTube processor
    yt_proc = active_youtube_processors.pop(session_id, None)
    if yt_proc:
        await yt_proc.stop()

    # Stop simulator if demo mode
    sim = active_simulators.pop(session_id, None)
    if sim:
        await sim.stop()

    # Remove live processor
    active_processors.pop(session_id, None)

    # Generate personalized notes using LLM if transcript exists
    personalized_notes = None
    if session.transcript:
        full_text = "\n".join([f"[{t.lecture_time}] {t.text}" for t in session.transcript])
        personalized_notes = await llm_service.generate_personalized_notes(full_text)

    # Auto-save notes to file
    notes_dir = os.path.join(os.path.dirname(__file__), "notes")
    saved_path = save_notes_to_file(session, personalized_notes, notes_dir)

    return {
        "session_id": session_id,
        "duration": session.duration_seconds,
        "notes_saved": saved_path,
        "llm_summary": "Generated" if personalized_notes else "Skipped"
    }


# ── Transcript endpoint (receives text from browser Speech API) ─
class TranscriptRequest(BaseModel):
    text: str
    speaker: str = "professor"

@app.post("/api/session/{session_id}/transcript")
async def receive_transcript(session_id: str, req: TranscriptRequest):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    processor = active_processors.get(session_id)
    if not processor:
        # Even in demo mode, allow transcript processing
        processor = LiveProcessor(send_callback=broadcast)
        active_processors[session_id] = processor

    result = await processor.process_transcript(req.text, session)
    return result


# ── Frame endpoint (receives base64 JPEG from camera) ───────────
class FrameRequest(BaseModel):
    frame: str

@app.post("/api/session/{session_id}/frame")
async def receive_frame(session_id: str, req: FrameRequest):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    processor = active_processors.get(session_id)
    if not processor:
        processor = LiveProcessor(send_callback=broadcast)
        active_processors[session_id] = processor

    result = await processor.process_frame(req.frame, session)
    return result


# ── Notes endpoints ──────────────────────────────────────────────
@app.get("/api/session/{session_id}/notes")
async def get_notes(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    markdown = generate_markdown_notes(session)
    return {"session_id": session_id, "markdown": markdown}


@app.get("/api/session/{session_id}/notes/download")
async def download_notes(session_id: str):
    """Download the saved notes file for a session."""
    notes_dir = os.path.join(os.path.dirname(__file__), "notes")
    # Find the most recent notes file for this session
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Try to find saved file
    import glob
    pattern = os.path.join(notes_dir, f"*{session_id[:8]}*")
    files = glob.glob(pattern)
    if not files:
        # Generate and save now
        saved_path = save_notes_to_file(session, notes_dir)
        if saved_path:
            return FileResponse(saved_path, filename=os.path.basename(saved_path), media_type="text/markdown")
        raise HTTPException(404, "No notes file found")

    return FileResponse(files[-1], filename=os.path.basename(files[-1]), media_type="text/markdown")


@app.get("/api/session/{session_id}/concepts")
async def get_concepts(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # For live mode, build graph from detected concepts
    if session.key_concepts:
        nodes = list({c.title for c in session.key_concepts})
        edges = []
        for c in session.key_concepts:
            for r in c.related_concepts:
                if r in nodes:
                    edges.append(ConceptLink(source=c.title, target=r, relationship="relates_to"))
        graph = ConceptGraph(nodes=nodes, edges=edges)
    else:
        # Fallback to demo graph
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
    print(f"Client connected ({len(connected_clients)} total)")
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
            # Handle transcript from client via WebSocket
            elif msg.get("type") == "transcript":
                for sid, session in active_sessions.items():
                    if session.is_active:
                        processor = active_processors.get(sid)
                        if not processor:
                            processor = LiveProcessor(send_callback=broadcast)
                            active_processors[sid] = processor
                        await processor.process_transcript(msg.get("text", ""), session)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)
        print(f"Client disconnected ({len(connected_clients)} total)")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
