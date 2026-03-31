import httpx
import json
import asyncio

class LLMService:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3" # Default model, can be configured

    async def generate_personalized_notes(self, transcript_text: str) -> str:
        """Use local LLM to generate professional, personalized notes."""
        prompt = f"""
        You are a professional teaching assistant. Below is a transcript from a lecture.
        Please transform this transcript into structured, detailed, and personalized study notes.
        Organize them by the main concepts explained, providing definitions, key takeaways, 
        and clear examples where mentioned.
        
        Transcript:
        {transcript_text}
        
        Professional Markdown Study Notes:
        """
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "LLM could not generate a response.")
                else:
                    return f"LLM Error: Status {response.status_code}. Make sure Ollama is running."
        except Exception as e:
            return f"LLM Connection Error: {str(e)}. Ensure Ollama is installed and running at {self.base_url}"

llm_service = LLMService()
