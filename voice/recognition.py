# voice/recognition.py - Voice recognition and identification
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity
from config import DEBUG, SAMPLE_RATE, VOICE_CONFIDENCE_THRESHOLD

# Initialize voice encoder
encoder = VoiceEncoder()

def generate_voice_embedding(audio):
    """Generate voice embedding from audio"""
    try:
        if len(audio) < SAMPLE_RATE:  # Need at least 1 second
            return None
        
        audio_float = audio.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0
        
        embedding = encoder.embed_utterance(audio_float)
        
        if embedding is not None and len(embedding) == 256:
            if DEBUG:
                print(f"[Buddy V2] ✅ Voice embedding generated")
            return embedding
        else:
            if DEBUG:
                print(f"[Buddy V2] ❌ Invalid embedding")
            return None
            
    except Exception as e:
        if DEBUG:
            print(f"[Buddy V2] Voice embedding error: {e}")
        return None

def identify_speaker(audio):
    """Identify speaker from audio - with proper unknown handling"""
    try:
        # Import known_users from database module
        from voice.database import known_users
        
        if not known_users:
            if DEBUG:
                print("[Buddy V2] 🆕 No voice profiles - needs training")
            return "UNKNOWN", 0.0
        
        embedding = generate_voice_embedding(audio)
        if embedding is None:
            if DEBUG:
                print("[Buddy V2] ❌ Could not generate voice embedding")
            return "UNKNOWN", 0.0
        
        best_match = None
        best_score = 0.0
        
        for name, stored_embedding in known_users.items():
            try:
                if isinstance(stored_embedding, list) and len(stored_embedding) == 256:
                    similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                    if DEBUG:
                        print(f"[Buddy V2] 🔍 {name}: {similarity:.3f}")
                    if similarity > best_score:
                        best_match = name
                        best_score = similarity
            except Exception as e:
                if DEBUG:
                    print(f"[Buddy V2] Comparison error for {name}: {e}")
                continue
        
        # Strict confidence threshold
        if best_match and best_score >= VOICE_CONFIDENCE_THRESHOLD:
            if DEBUG:
                print(f"[Buddy V2] ✅ HIGH CONFIDENCE: {best_match} ({best_score:.3f})")
            return best_match, best_score
        else:
            if DEBUG:
                print(f"[Buddy V2] ❓ LOW CONFIDENCE: best={best_match}, score={best_score:.3f}")
            return "UNKNOWN", best_score
            
    except Exception as e:
        if DEBUG:
            print(f"[Buddy V2] Speaker identification error: {e}")
        return "UNKNOWN", 0.0

def register_voice(audio, username):
    """Register a new voice"""
    try:
        # Import database functions
        from voice.database import known_users, save_known_users
        
        embedding = generate_voice_embedding(audio)
        if embedding is None:
            return False
        
        known_users[username] = embedding.tolist()
        save_known_users()
        
        print(f"[Buddy V2] 🎊 Registered voice for {username}")
        return True
        
    except Exception as e:
        print(f"[Buddy V2] Voice registration error: {e}")
        return False