# audio/processing.py - Audio processing utilities
import numpy as np
from scipy.signal import resample_poly
from config import SAMPLE_RATE

def downsample_audio(audio, orig_sr, target_sr):
    """Downsample audio to target sample rate"""
    if audio.ndim > 1:
        audio = audio[:, 0]  # ensure mono
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    if orig_sr != target_sr:
        audio = resample_poly(audio, target_sr, orig_sr)
    
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)

def is_noise_or_gibberish(text):
    """Filter out noise and gibberish"""
    if not text or len(text.strip()) < 2:
        return True
    words = text.strip().split()
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    if len(words) < 2 and avg_len < 4:
        return True
    return False