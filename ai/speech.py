# ai/speech.py - Speech-to-text processing
import asyncio
import json
import numpy as np
import websockets
from config import FASTER_WHISPER_WS, DEBUG

async def whisper_stt_async(audio):
    """Transcribe audio using Whisper WebSocket"""
    try:
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        
        async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
            await ws.send(audio.tobytes())
            await ws.send("end")
            
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                print("[Buddy V2] Whisper timeout")
                return ""
            
            try:
                data = json.loads(message)
                text = data.get("text", "").strip()
                if DEBUG:
                    print(f"[Buddy V2] 📝 Whisper: '{text}'")
                return text
            except:
                text = message.decode("utf-8") if isinstance(message, bytes) else message
                if DEBUG:
                    print(f"[Buddy V2] 📝 Whisper: '{text}'")
                return text.strip()
                
    except Exception as e:
        print(f"[Buddy V2] Whisper error: {e}")
        return ""

def transcribe_audio(audio):
    """Synchronous wrapper for Whisper STT"""
    return asyncio.run(whisper_stt_async(audio))