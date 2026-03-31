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
            
            # 2. Get Video URL for snapshots (using yt-dlp)
            ydl_opts = {
                'format': 'bestvideo[height<=480]',
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info['url']

            # 3. Start streaming events
            # We'll run transcript and frame extraction in parallel
            await asyncio.gather(
                self._stream_transcript(transcript_list, session),
                self._stream_frames(video_url, session)
            )
            
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            await self.send(WSMessage(
                event_type=EventType.ALERT,
                data={"message": f"YouTube Error: {str(e)}", "alert_type": "error", "lecture_time": "00:00:00"}
            ))
        finally:
            self.running = False

    async def _stream_transcript(self, transcript_list, session: LectureSession):
        for entry in transcript_list:
            if not self.running:
                break
            
            wait_time = entry['start'] - (time.time() - self.start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            t_event = TranscriptionEvent(
                text=entry['text'],
                speaker=SpeakerRole.PROFESSOR,
                confidence=0.95,
                lecture_time=self._format_time(entry['start']),
            )
            # session.transcript.append(t_event) # Removed: _store_event handles this via callback
            await self.send(WSMessage(
                event_type=EventType.TRANSCRIPTION,
                data=t_event.model_dump()
            ))

    async def _stream_frames(self, video_url: str, session: LectureSession):
        """Extract frames every 60 seconds to simulate slides."""
        import subprocess
        
        slide_count = 0
        while self.running:
            elapsed = time.time() - self.start_time
            lt = self._format_time(elapsed)
            
            # Extract frame using ffmpeg
            # -ss seeks to time, -i is input, -vframes 1 captures one frame, -f image2 pipes to stdout
            cmd = [
                'ffmpeg', '-y', '-ss', str(int(elapsed)), '-i', video_url,
                '-vframes', '1', '-q:v', '2', '-f', 'image2pipe', '-'
            ]
            
            try:
                # Run ffmpeg sparingly (e.g., every 60s)
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if stdout:
                    import base64
                    frame_b64 = base64.b64encode(stdout).decode('utf-8')
                    slide_count += 1
                    s_event = SlideEvent(
                        slide_number=slide_count,
                        title=f"YouTube Snapshot at {lt}",
                        content_text="[Captured from video stream]",
                        lecture_time=lt,
                        snapshot_url=f"data:image/jpeg;base64,{frame_b64}"
                    )
                    # session.slides.append(s_event)
                    await self.send(WSMessage(
                        event_type=EventType.SLIDE_CHANGE,
                        data=s_event.model_dump()
                    ))
            except Exception as e:
                print(f"Frame extraction failed: {e}")

            # Wait for next snapshot
            await asyncio.sleep(60)

    def _format_time(self, seconds: float) -> str:
        s = int(seconds)
        h, m, s = s // 3600, (s % 3600) // 60, s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    async def stop(self):
        self.running = False
