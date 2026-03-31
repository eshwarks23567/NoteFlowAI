"""YouTube Processor — fetches transcript and frames from YouTube for analysis."""
import os, time, re, asyncio
from typing import Callable, Awaitable, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from models.schemas import (
    TranscriptionEvent, SpeakerRole, EventType, WSMessage, SlideEvent, LectureSession
)

class YoutubeProcessor:
    def __init__(self, send_callback: Callable[[WSMessage], Awaitable[None]]):
        self.send = send_callback
        self.running = False
        self.start_time = 0.0
        
    def _extract_video_id(self, url: str) -> str:
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else url

    async def process_video(self, url: str, session: LectureSession):
        video_id = self._extract_video_id(url)
        self.running = True
        self.start_time = time.time()

        try:
            # 1. Get Transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # 2. Start streaming events
            for entry in transcript_list:
                if not self.running:
                    break
                
                # Calculate wait time to simulate live lecture
                wait_time = entry['start'] - (time.time() - self.start_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Send transcript event
                t_event = TranscriptionEvent(
                    text=entry['text'],
                    speaker=SpeakerRole.PROFESSOR,
                    confidence=0.95,
                    is_emphasis_phrase=False, 
                    keywords=[],
                    lecture_time=self._format_time(entry['start']),
                )
                
                # Push to session history
                session.transcript.append(t_event)
                
                # Broadcast
                await self.send(WSMessage(
                    event_type=EventType.TRANSCRIPTION,
                    data=t_event.model_dump()
                ))
            
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            await self.send(WSMessage(
                event_type=EventType.ALERT,
                data={"message": f"YouTube Error: {str(e)}", "alert_type": "error"}
            ))
        finally:
            self.running = False

    def _format_time(self, seconds: float) -> str:
        s = int(seconds)
        h, m, s = s // 3600, (s % 3600) // 60, s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    async def stop(self):
        self.running = False
