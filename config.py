"""
Buddy Voice Assistant Configuration
Updated: 2025-07-03 09:10:39
"""
import os

# ==== SYSTEM INFORMATION ====
CURRENT_TIMESTAMP = "2025-07-03 09:10:39"  # Updated to current time
SYSTEM_USER = "Daveydrz"

# ==== DEBUG SETTINGS ====
DEBUG = True

# ==== LANGUAGE SETTINGS ====
DEFAULT_LANG = "en"

# ==== AUDIO DEVICE CONFIGURATION ====
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
CHANNELS = 1

# ==== FILE PATHS ====
KNOWN_USERS_PATH = "known_users_v2.json"
CONVERSATION_HISTORY_PATH = "conversation_history_v2.json"
CHIME_PATH = "chime.wav"

# ==== KOKORO TTS MODEL PATHS ====
KOKORO_MODEL_PATH = "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "voices-v1.0.bin"

# ==== WEBSOCKET URLS ====
FASTER_WHISPER_WS = "ws://localhost:9090"

# ==== KOKORO TTS CONFIGURATION ====
KOKORO_VOICES = {"pl": "af_heart", "en": "af_heart", "it": "if_sara"}
KOKORO_LANGS = {"pl": "pl", "en": "en-us", "it": "it"}

# ==== WAKE WORD CONFIGURATION ====
PORCUPINE_ACCESS_KEY = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
WAKE_WORD_PATH = r"hey-buddy_en_windows_v3_0_0.ppn"

# ==== VOICE RECOGNITION SETTINGS ====
VOICE_CONFIDENCE_THRESHOLD = 0.70
VOICE_EMBEDDING_DIM = 256

# ==== CONVERSATION SETTINGS ====
MAX_HISTORY_LENGTH = 6

# ==== TRAINING PHRASES ====
TRAINING_PHRASES = [
    "Hey Buddy, how are you?",
    "What's the weather like?", 
    "Can you help me?",
    "Tell me something cool.",
    "This is my voice."
]

# ==== LLM SETTINGS ====
KOBOLD_URL = "http://localhost:5001/v1/chat/completions"
MAX_TOKENS = 80
TEMPERATURE = 0.7

# ==== VOICE TRAINING MODES ====
TRAINING_MODE_NONE = 0
TRAINING_MODE_FORMAL = 1
TRAINING_MODE_PASSIVE = 2

# ==== FULL DUPLEX SETTINGS ====
FULL_DUPLEX_MODE = True
INTERRUPT_DETECTION = True
CONTINUOUS_LISTENING = True
REAL_TIME_PROCESSING = True

# ==== VOICE ACTIVITY DETECTION ====
# Note: Optimized for better speech detection (less aggressive filtering)
VAD_THRESHOLD = 400                 # Lower threshold for better sensitivity
MIN_SPEECH_FRAMES = 4               # Faster speech detection
MAX_SILENCE_FRAMES = 150            # More patience for pauses
INTERRUPT_THRESHOLD = 1500
MIN_SPEECH_DURATION = 0.2           # Accept shorter speech segments
SPEECH_PADDING_START = 0.8          # More context before speech
SPEECH_PADDING_END = 0.8            # More context after speech
WHISPER_CONTEXT_PADDING = 3.0       # Maximum context for Whisper

# ==== VOICE FINGERPRINTING SETTINGS ====
# Note: Disabled for testing to avoid speech filtering conflicts
VOICE_FINGERPRINTING = False        # DISABLED for testing
BUDDY_VOICE_THRESHOLD = 0.99        # Nearly impossible to trigger when enabled
VOICE_SIMILARITY_BUFFER = 5         # Number of samples for consistency check
VOICE_LEARNING_PATIENCE = 10        # Number of samples for learning

# ==== ACOUSTIC ECHO CANCELLATION (AEC) SETTINGS ====
AEC_ENABLED = True
AEC_AGGRESSIVE_MODE = False         # Conservative mode for better quality
AEC_CONSERVATIVE_MODE = True
AEC_ADAPTATION_RATE = 0.1
AEC_SUPPRESSION_FACTOR = 0.3
AEC_VOICE_PROTECTION = True

# ==== HUMAN SPEECH PROTECTION ====
# Note: Disabled for testing to avoid conflicts with voice processing
HUMAN_SPEECH_PROTECTION = True     # DISABLED for testing
MINIMAL_PROCESSING_MODE = True
SPEECH_QUALITY_PRIORITY = True
