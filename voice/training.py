# voice/training.py - Voice training system
import re
import time
import numpy as np
from audio.input import aec_training_listen
from audio.output import speak_async, play_chime, buddy_talking
from ai.speech import transcribe_audio
from voice.recognition import generate_voice_embedding
from config import *

def voice_training_mode():
    """Voice training with PROPER AEC timing - FIXED"""
    print("\n" + "="*50)
    print("🎓 BUDDY VOICE TRAINING - PROPER AEC")
    print("="*50)
    
    # Get user's name with MUCH better timing
    speak_async("Hi! I need to learn your voice. When you hear the beep, say only your first name.", DEFAULT_LANG)
    
    # Wait for TTS to finish completely
    while buddy_talking.is_set():
        time.sleep(0.1)
    
    # CRITICAL: Wait much longer for AEC to fully clear Buddy's voice
    print("[Training] ⏳ Waiting for audio to clear...")
    time.sleep(5.0)  # Much longer wait for complete audio clearing
    
    # Play a beep to signal when to talk
    play_chime()
    time.sleep(0.5)
    
    print("[Training] 👂 NOW listening for your name only...")
    
    # Get name with proper AEC timing
    name_audio = aec_training_listen("name", timeout=8)
    if name_audio is None:
        speak_async("I didn't hear your name. Let's try again later.", DEFAULT_LANG)
        return False
    
    # Transcribe name
    name_text = transcribe_audio(name_audio)
    if not name_text or len(name_text.strip()) < 2:
        speak_async("I couldn't understand your name. Let's try again later.", DEFAULT_LANG)
        return False
    
    print(f"[Training] 📝 Raw transcription: '{name_text}'")
    
    # MUCH better name extraction - only look for single words
    username = extract_single_name(name_text)
    if not username:
        # Fallback to Daveydrz if extraction fails
        username = SYSTEM_USER
        print(f"[Training] 🔄 Using fallback: {username}")
    
    print(f"[Training] 👤 Training voice for: {username}")
    
    # Confirm and proceed with training
    speak_async(f"Got it, {username}! Now I'll teach you 5 phrases. Listen carefully.", DEFAULT_LANG)
    
    while buddy_talking.is_set():
        time.sleep(0.1)
    time.sleep(5.0)  # Long pause for complete audio clearing
    
    voice_samples = []
    successful_recordings = 0
    
    for i, phrase in enumerate(TRAINING_PHRASES):
        print(f"\n[Training] 📝 Phrase {i+1}/{len(TRAINING_PHRASES)}: {phrase}")
        
        # Say the phrase
        speak_async(f"Phrase {i+1}: {phrase}", DEFAULT_LANG)
        while buddy_talking.is_set():
            time.sleep(0.1)
        
        # Critical: Long pause for AEC clearing
        time.sleep(4.0)
        
        # Signal when to speak
        speak_async("Now you say it.", DEFAULT_LANG)
        while buddy_talking.is_set():
            time.sleep(0.1)
        
        # Even longer pause for complete clearing
        time.sleep(5.0)
        
        # Beep to signal recording start
        play_chime()
        time.sleep(0.5)
        
        print(f"[Training] 🔴 RECORDING NOW: '{phrase}'")
        
        # Record with proper AEC
        audio = aec_training_listen(f"phrase_{i+1}", timeout=10)
        
        if audio is not None and len(audio) > SAMPLE_RATE * 2:
            voice_samples.append(audio)
            successful_recordings += 1
            print(f"[Training] ✅ Got phrase {i+1}")
            speak_async("Perfect!", DEFAULT_LANG)
        else:
            print(f"[Training] ❌ Failed phrase {i+1}")
            speak_async("Let's try that again.", DEFAULT_LANG)
            
            while buddy_talking.is_set():
                time.sleep(0.1)
            time.sleep(5.0)
            
            play_chime()
            time.sleep(0.5)
            
            # Retry
            audio = aec_training_listen(f"phrase_{i+1}_retry", timeout=10)
            if audio is not None and len(audio) > SAMPLE_RATE * 2:
                voice_samples.append(audio)
                successful_recordings += 1
                speak_async("Got it!", DEFAULT_LANG)
            else:
                speak_async("We'll skip that one.", DEFAULT_LANG)
        
        # Pause between phrases
        while buddy_talking.is_set():
            time.sleep(0.1)
        time.sleep(2.0)
    
    # Process results
    if successful_recordings >= 3:
        print(f"[Training] 🧠 Processing {successful_recordings} samples...")
        speak_async("Processing your voice now.", DEFAULT_LANG)
        
        # Generate embeddings
        embeddings = []
        for sample in voice_samples:
            embedding = generate_voice_embedding(sample)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) >= 2:
            # Import database functions INSIDE the function
            from voice.database import known_users, save_known_users
            
            # Save voice profile
            avg_embedding = np.mean(embeddings, axis=0)
            known_users[username] = avg_embedding.tolist()
            save_known_users()
            
            print(f"[Training] 🎊 SUCCESS! Voice learned for {username}")
            speak_async(f"Perfect! I've learned your voice, {username}. I'll recognize you now!", DEFAULT_LANG)
            
            return True
        else:
            speak_async("I had trouble processing your voice. Let's try again later.", DEFAULT_LANG)
            return False
    else:
        speak_async(f"I only got {successful_recordings} samples. I need at least 3. Let's try again later.", DEFAULT_LANG)
        return False

def extract_single_name(text):
    """Extract single name from text - IMPROVED for training"""
    try:
        print(f"[Training] 🔍 Looking for single name in: '{text}'")
        
        # Split into words and look for the last standalone name
        words = text.lower().split()
        
        # Look for common name patterns at the end
        if len(words) >= 1:
            last_word = words[-1].replace('!', '').replace('.', '').replace('?', '')
            if last_word.isalpha() and len(last_word) >= 3:
                # Check if it's a common name, not an instruction word
                instruction_words = ['voice', 'name', 'first', 'clearly', 'example', 'sarah', 'say', 'stop', 'talking', 'when', 'your', 'only', 'just']
                if last_word not in instruction_words:
                    print(f"[Training] ✅ Found name: '{last_word.title()}'")
                    return last_word.title()
        
        # Look for capitalized words that aren't common instruction words
        for word in reversed(words):  # Check from end backwards
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if (clean_word.isalpha() and 
                len(clean_word) >= 3 and
                clean_word.lower() not in ['voice', 'name', 'first', 'clearly', 'example', 'sarah', 'david', 'say', 'stop', 'talking', 'when', 'your', 'only', 'just', 'learn', 'need', 'finish']):
                print(f"[Training] ✅ Found name: '{clean_word.title()}'")
                return clean_word.title()
        
        print(f"[Training] ❌ No valid name found")
        return None
        
    except Exception as e:
        print(f"[Training] Name extraction error: {e}")
        return None

def check_voice_training_command(text):
    """Check if user wants to train their voice - EXPANDED"""
    training_commands = [
        "learn my voice",
        "train my voice", 
        "remember my voice",
        "voice training",
        "teach you my voice",
        "register my voice",
        "retrain my voice",
        "fix my voice"
    ]
    
    text_lower = text.lower().strip()
    return any(cmd in text_lower for cmd in training_commands)