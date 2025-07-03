# audio/voice_fingerprint.py - Fixed with safe config loading
import numpy as np
import time
from resemblyzer import VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

# Safe config loading
try:
    from config import DEBUG, BUDDY_VOICE_THRESHOLD, VOICE_SIMILARITY_BUFFER, VOICE_LEARNING_PATIENCE
except ImportError:
    DEBUG = True
    BUDDY_VOICE_THRESHOLD = 0.90
    VOICE_SIMILARITY_BUFFER = 5
    VOICE_LEARNING_PATIENCE = 10
    print("[VoiceFingerprint] ⚠️ Using default voice fingerprint settings")

# Initialize voice encoder for fingerprinting
try:
    encoder = VoiceEncoder()
    print("[VoiceFingerprint] ✅ Voice encoder loaded")
except Exception as e:
    print(f"[VoiceFingerprint] ❌ Error loading encoder: {e}")
    encoder = None

# Buddy's voice profile (learned from TTS)
buddy_voice_profile = None
buddy_voice_samples = []
buddy_voice_threshold = BUDDY_VOICE_THRESHOLD
buddy_voice_buffer = deque(maxlen=VOICE_SIMILARITY_BUFFER)

def learn_buddy_voice(audio_samples):
    """Learn Buddy's voice from TTS samples"""
    global buddy_voice_profile, buddy_voice_samples
    
    if encoder is None:
        return False
    
    try:
        print("[VoiceFingerprint] 🎓 Learning Buddy's voice...")
        
        # Process multiple TTS samples
        embeddings = []
        for sample in audio_samples:
            if len(sample) >= 16000:  # At least 1 second
                audio_float = sample.astype(np.float32) / 32768.0
                embedding = encoder.embed_utterance(audio_float)
                if embedding is not None:
                    embeddings.append(embedding)
        
        if len(embeddings) >= 3:
            # Create average voice profile
            buddy_voice_profile = np.mean(embeddings, axis=0)
            buddy_voice_samples = audio_samples.copy()
            print(f"[VoiceFingerprint] ✅ Buddy's voice learned from {len(embeddings)} samples")
            return True
        else:
            print(f"[VoiceFingerprint] ❌ Not enough samples to learn voice")
            return False
            
    except Exception as e:
        print(f"[VoiceFingerprint] ❌ Error learning voice: {e}")
        return False

def is_buddy_speaking(audio_chunk):
    """CONSERVATIVE check if audio chunk is Buddy's voice"""
    global buddy_voice_profile, buddy_voice_buffer
    
    if encoder is None or buddy_voice_profile is None:
        return False
    
    try:
        if len(audio_chunk) < 8000:  # Need at least 0.5 seconds
            return False
        
        # Generate embedding for this chunk
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        embedding = encoder.embed_utterance(audio_float)
        
        if embedding is not None:
            # Compare with Buddy's voice profile
            similarity = cosine_similarity([embedding], [buddy_voice_profile])[0][0]
            
            # Add to buffer for consistency check
            buddy_voice_buffer.append(similarity)
            
            if DEBUG and similarity > 0.6:
                print(f"[VoiceFingerprint] 🔍 Voice similarity: {similarity:.3f}")
            
            # CONSERVATIVE decision - require high similarity AND consistency
            if len(buddy_voice_buffer) >= 3:
                recent_similarities = list(buddy_voice_buffer)[-3:]
                avg_similarity = np.mean(recent_similarities)
                
                # Only reject if VERY sure it's Buddy's voice
                is_buddy = (avg_similarity >= buddy_voice_threshold and 
                           similarity >= buddy_voice_threshold * 0.95)
                
                if is_buddy and DEBUG:
                    print(f"[VoiceFingerprint] 🤖 BUDDY VOICE DETECTED (avg:{avg_similarity:.3f})")
                
                return is_buddy
            else:
                # Not enough samples yet
                return False
        
    except Exception as e:
        if DEBUG:
            print(f"[VoiceFingerprint] Error checking voice: {e}")
    
    return False

def add_buddy_sample(audio_sample):
    """Add a new Buddy TTS sample for learning"""
    global buddy_voice_samples
    
    try:
        if len(buddy_voice_samples) < VOICE_LEARNING_PATIENCE:
            buddy_voice_samples.append(audio_sample)
            
            # Re-learn voice profile with new sample
            if len(buddy_voice_samples) >= 5:
                learn_buddy_voice(buddy_voice_samples)
        else:
            # Replace oldest sample
            buddy_voice_samples.pop(0)
            buddy_voice_samples.append(audio_sample)
            learn_buddy_voice(buddy_voice_samples)
            
    except Exception as e:
        if DEBUG:
            print(f"[VoiceFingerprint] Error adding sample: {e}")

def get_buddy_voice_status():
    """Get status of Buddy's voice learning"""
    return {
        "learned": buddy_voice_profile is not None,
        "samples": len(buddy_voice_samples),
        "threshold": buddy_voice_threshold,
        "buffer_size": len(buddy_voice_buffer)
    }

def reset_voice_learning():
    """Reset voice learning (for debugging)"""
    global buddy_voice_profile, buddy_voice_samples, buddy_voice_buffer
    buddy_voice_profile = None
    buddy_voice_samples = []
    buddy_voice_buffer.clear()
    print("[VoiceFingerprint] 🔄 Voice learning reset")