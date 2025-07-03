# voice/manager.py - Smart voice management system
import time
from voice.database import known_users, save_known_users
from voice.recognition import identify_speaker, generate_voice_embedding
from voice.training import voice_training_mode
from audio.output import speak_async
from ai.speech import transcribe_audio
from config import *

class VoiceManager:
    def __init__(self):
        self.unknown_speakers = {}  # Track unknown speakers for passive training
        self.training_mode = TRAINING_MODE_NONE
        self.current_training_user = None
        self.passive_samples = []
        
    def handle_voice_identification(self, audio, text):
        """Smart voice identification and training handler"""
        
        # First, try to identify the speaker
        identified_user, confidence = identify_speaker(audio)
        
        if identified_user != "UNKNOWN" and confidence >= VOICE_CONFIDENCE_THRESHOLD:
            # Known user with high confidence
            print(f"[Voice] ✅ Recognized: {identified_user} (confidence: {confidence:.3f})")
            return identified_user, "RECOGNIZED"
        
        elif identified_user != "UNKNOWN" and confidence >= 0.4:
            # Known user but lower confidence - still recognize but mention
            print(f"[Voice] 🤔 Probably: {identified_user} (confidence: {confidence:.3f})")
            return identified_user, "LIKELY"
        
        else:
            # Unknown speaker - handle gracefully
            print(f"[Voice] ❓ Unknown speaker (best: {identified_user}, confidence: {confidence:.3f})")
            return self.handle_unknown_speaker(audio, text)
    
    def handle_unknown_speaker(self, audio, text):
        """Handle unknown speaker intelligently"""
        
        # Check if this is a voice training command
        if self.is_voice_training_command(text):
            print("[Voice] 🎓 Voice training requested by unknown speaker")
            success = voice_training_mode()
            if success:
                return SYSTEM_USER, "TRAINED"
            else:
                return "Guest", "TRAINING_FAILED"
        
        # Check if we have ANY voice profiles
        if not known_users:
            print("[Voice] 🆕 No voice profiles exist - offering training")
            self.offer_voice_training()
            return "Guest", "NEEDS_TRAINING"
        
        # We have profiles but don't recognize this person
        print("[Voice] 👤 Unknown speaker with existing profiles")
        self.handle_unrecognized_speaker(audio, text)
        return "Guest", "UNRECOGNIZED"
    
    def offer_voice_training(self):
        """Offer voice training to new user"""
        speak_async("I don't recognize your voice. Would you like me to learn it so I can identify you in the future? Just say yes or no.", DEFAULT_LANG)
    
    def handle_unrecognized_speaker(self, audio, text):
        """Handle speaker we don't recognize"""
        speak_async("I don't recognize your voice. What's your name?", DEFAULT_LANG)
        # The main loop will handle the response
    
    def is_voice_training_command(self, text):
        """Check if user wants voice training"""
        training_commands = [
            "learn my voice", "train my voice", "remember my voice",
            "voice training", "teach you my voice", "register my voice",
            "yes", "yeah", "sure", "okay", "ok"
        ]
        text_lower = text.lower().strip()
        return any(cmd in text_lower for cmd in training_commands)
    
    def is_training_decline(self, text):
        """Check if user declined training"""
        decline_words = ["no", "nope", "not now", "later", "skip", "maybe later"]
        text_lower = text.lower().strip()
        return any(word in text_lower for word in decline_words)
    
    def start_passive_training(self, username):
        """Start passive voice training during conversation"""
        print(f"[Voice] 🎯 Starting passive training for {username}")
        self.training_mode = TRAINING_MODE_PASSIVE
        self.current_training_user = username
        self.passive_samples = []
        speak_async(f"Okay {username}, I'll try to learn your voice as we talk. Just speak naturally!", DEFAULT_LANG)
    
    def add_passive_sample(self, audio, text):
        """Add a passive training sample"""
        if self.training_mode == TRAINING_MODE_PASSIVE and len(text.split()) >= 3:
            # Only use longer utterances for passive training
            self.passive_samples.append(audio)
            print(f"[Voice] 📊 Passive sample added ({len(self.passive_samples)}/10)")
            
            # Once we have enough samples, try to create a profile
            if len(self.passive_samples) >= 10:
                self.finalize_passive_training()
    
    def finalize_passive_training(self):
        """Finalize passive training with collected samples"""
        try:
            print(f"[Voice] 🧠 Finalizing passive training for {self.current_training_user}")
            
            # Generate embeddings from samples
            embeddings = []
            for sample in self.passive_samples:
                embedding = generate_voice_embedding(sample)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if len(embeddings) >= 5:
                # Create average embedding
                import numpy as np
                avg_embedding = np.mean(embeddings, axis=0)
                known_users[self.current_training_user] = avg_embedding.tolist()
                save_known_users()
                
                print(f"[Voice] 🎊 Passive training successful for {self.current_training_user}")
                speak_async(f"Great! I've learned your voice, {self.current_training_user}. I should recognize you better now!", DEFAULT_LANG)
                
                # Reset training mode
                self.training_mode = TRAINING_MODE_NONE
                self.current_training_user = None
                self.passive_samples = []
                return True
            else:
                print(f"[Voice] ❌ Not enough good samples for passive training")
                return False
                
        except Exception as e:
            print(f"[Voice] ❌ Passive training error: {e}")
            return False

# Global voice manager instance
voice_manager = VoiceManager()