# audio/output.py - Full Duplex Audio Output
import threading
import time
import queue
import numpy as np
import simpleaudio as sa
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly
from langdetect import detect
from config import *

# Global audio state
audio_queue = queue.Queue()
current_audio_playback = None
audio_lock = threading.Lock()
buddy_talking = threading.Event()
playback_start_time = None

# Initialize Kokoro TTS
try:
    kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    print("[Buddy V2] ✅ Kokoro TTS loaded successfully")
except FileNotFoundError as e:
    print(f"[Buddy V2] ❌ Kokoro files missing: {e}")
    kokoro = None

def generate_tts(text, lang=DEFAULT_LANG):
    """Generate TTS audio using Kokoro"""
    try:
        if kokoro is None:
            return None, None
            
        detected_lang = lang or detect(text)
        voice = KOKORO_VOICES.get(detected_lang, KOKORO_VOICES["en"])
        kokoro_lang = KOKORO_LANGS.get(detected_lang, "en-us")

        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)

        if len(samples) == 0:
            return None, None

        samples_16k = resample_poly(samples, SAMPLE_RATE, sample_rate)
        samples_16k = np.clip(samples_16k, -1.0, 1.0)
        pcm_16k = (samples_16k * 32767).astype(np.int16)

        if DEBUG:
            print(f"[Buddy V2] 🗣️ Generated TTS: {len(pcm_16k)} samples")
        return pcm_16k, SAMPLE_RATE
        
    except Exception as e:
        print(f"[Buddy V2] TTS error: {e}")
        return None, None

def speak_async(text, lang=DEFAULT_LANG):
    """Queue text for speech synthesis"""
    if not text or len(text.strip()) < 2:
        return
        
    def tts_worker():
        pcm, sr = generate_tts(text.strip(), lang)
        if pcm is not None:
            audio_queue.put((pcm, sr))
    
    threading.Thread(target=tts_worker, daemon=True).start()

def play_chime():
    """Play notification chime"""
    try:
        from pydub import AudioSegment
        from audio.processing import downsample_audio
        
        audio = AudioSegment.from_wav(CHIME_PATH)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples[:, 0]
        
        if audio.frame_rate != SAMPLE_RATE:
            samples = downsample_audio(samples, audio.frame_rate, SAMPLE_RATE)
        
        audio_queue.put((samples, SAMPLE_RATE))
    except Exception as e:
        if DEBUG:
            print(f"[Buddy V2] Chime error: {e}")

def audio_worker():
    """Full Duplex Audio Worker"""
    global current_audio_playback, playback_start_time
    
    mode_name = "FULL DUPLEX" if FULL_DUPLEX_MODE else "HALF DUPLEX"
    print(f"[Buddy V2] 🎵 {mode_name} Audio Worker started")
    
    while True:
        try:
            item = audio_queue.get()
            if item is None:
                break
                
            pcm, sr = item
            
            with audio_lock:
                # Set buddy talking state (only for half duplex)
                if not FULL_DUPLEX_MODE:
                    buddy_talking.set()
                
                playback_start_time = time.time()
                
                # Notify full duplex manager if active
                if FULL_DUPLEX_MODE:
                    from audio.full_duplex_manager import full_duplex_manager
                    full_duplex_manager.notify_buddy_speaking(pcm)
                
                # Play audio
                current_audio_playback = sa.play_buffer(pcm.tobytes(), 1, 2, sr)
                
                if DEBUG:
                    print(f"[Audio] 🎵 Playing in {mode_name} mode")
                
                # Wait for completion
                while current_audio_playback and current_audio_playback.is_playing():
                    time.sleep(0.01)
                
                # Cleanup
                current_audio_playback = None
                
                # Half duplex cleanup
                if not FULL_DUPLEX_MODE:
                    time.sleep(0.3)
                    buddy_talking.clear()
                
                playback_start_time = None
                
            audio_queue.task_done()
            
        except Exception as e:
            print(f"[Buddy V2] Audio worker error: {e}")
            if not FULL_DUPLEX_MODE:
                buddy_talking.clear()
            playback_start_time = None
            current_audio_playback = None

def start_audio_worker():
    """Start the audio worker thread"""
    threading.Thread(target=audio_worker, daemon=True).start()

def is_buddy_talking():
    """Check if Buddy is currently talking"""
    if FULL_DUPLEX_MODE:
        return current_audio_playback is not None and current_audio_playback.is_playing()
    else:
        return buddy_talking.is_set()