# config.py - All configuration settings
import os

# System Information
CURRENT_TIMESTAMP = "2025-07-02 11:13:09"
SYSTEM_USER = "Daveydrz"

# Debug Settings
DEBUG = True

# Language Settings
DEFAULT_LANG = "en"

# Audio Device Configuration
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
CHANNELS = 1

# File Paths
KNOWN_USERS_PATH = "known_users_v2.json"
CONVERSATION_HISTORY_PATH = "conversation_history_v2.json"
CHIME_PATH = "chime.wav"

# Kokoro model paths
KOKORO_MODEL_PATH = "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "voices-v1.0.bin"

# WebSocket URLs
FASTER_WHISPER_WS = "ws://localhost:9090"

# Kokoro TTS Configuration
KOKORO_VOICES = {"pl": "af_heart", "en": "af_heart", "it": "if_sara"}
KOKORO_LANGS = {"pl": "pl", "en": "en-us", "it": "it"}

# Wake Word Configuration
PORCUPINE_ACCESS_KEY = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
WAKE_WORD_PATH = r"hey-buddy_en_windows_v3_0_0.ppn"

# Voice Recognition Settings
VOICE_CONFIDENCE_THRESHOLD = 0.70
VOICE_EMBEDDING_DIM = 256

# Conversation Settings
MAX_HISTORY_LENGTH = 6

# Training Phrases
TRAINING_PHRASES = [
    "Hey Buddy, how are you?",
    "What's the weather like?", 
    "Can you help me?",
    "Tell me something cool.",
    "This is my voice."
]

# LLM Settings
KOBOLD_URL = "http://localhost:5001/v1/chat/completions"
MAX_TOKENS = 80
TEMPERATURE = 0.7

# Voice training modes
TRAINING_MODE_NONE = 0
TRAINING_MODE_FORMAL = 1
TRAINING_MODE_PASSIVE = 2

# ==== FULL DUPLEX SETTINGS ====
FULL_DUPLEX_MODE = True  # 🚀 ENABLED!
AEC_AGGRESSIVE_MODE = True
VOICE_FINGERPRINTING = True
INTERRUPT_DETECTION = True
CONTINUOUS_LISTENING = True
REAL_TIME_PROCESSING = True

# ==== IMPROVED SPEECH CAPTURE SETTINGS ====
VAD_THRESHOLD = 800               # LOWERED - More sensitive to speech
MIN_SPEECH_FRAMES = 8             # LOWERED - Faster speech detection  
MAX_SILENCE_FRAMES = 100          # INCREASED - Allow longer pauses
INTERRUPT_THRESHOLD = 1500        # Keep high for interrupts
MIN_SPEECH_DURATION = 0.5         # LOWERED - Accept shorter speech
SPEECH_PADDING_START = 0.3        # Add 300ms before detected speech
SPEECH_PADDING_END = 0.5          # Add 500ms after speech ends
WHISPER_CONTEXT_PADDING = 1.0     # Extra context for Whisper