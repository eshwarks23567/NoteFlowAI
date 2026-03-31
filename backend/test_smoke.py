"""Smoke test for the live processor and note generator."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.live_processor import LiveProcessor
from services.note_generator import generate_markdown_notes, save_notes_to_file
from models.schemas import LectureSession


async def test():
    lp = LiveProcessor()
    session = LectureSession(title="Test Lecture")

    # Test transcript processing
    result = await lp.process_transcript(
        "This is important. Gradient descent is the backbone of neural networks. "
        "Remember that regularization prevents overfitting.",
        session,
    )
    print(f"Transcript events: {len(session.transcript)}")
    print(f"Importance events: {len(session.importance_events)}")
    print(f"Key concepts: {[c.title for c in session.key_concepts]}")
    print(f"Result: {result}")

    assert len(session.transcript) > 0, "No transcript events!"
    assert result["status"] == "processed", f"Status was {result['status']}"
    print("✅ Transcript processing: PASS")

    # Test note generation
    md = generate_markdown_notes(session)
    assert len(md) > 50, "Notes too short!"
    assert "Test Lecture" in md, "Title missing from notes!"
    print(f"✅ Note generation: PASS ({len(md)} chars)")

    # Test save
    os.makedirs("notes", exist_ok=True)
    saved = save_notes_to_file(session, "notes")
    assert saved is not None, "Save returned None!"
    assert os.path.exists(saved), f"File not found: {saved}"
    print(f"✅ Notes auto-save: PASS ({saved})")

    # Clean up test file
    os.remove(saved)

    print("\n🎉 ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(test())
