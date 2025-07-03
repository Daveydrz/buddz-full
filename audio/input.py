# audio/input.py - Full Duplex Ready Audio Input
import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
from audio.processing import downsample_audio
from audio.output import buddy_talking, is_buddy_talking
from config import *

def simple_vad_listen():
    """Full Duplex Ready Voice Activity Detection"""
    if FULL_DUPLEX_MODE:
        return full_duplex_vad_listen()
    else:
        return half_duplex_vad_listen()

def full_duplex_vad_listen():
    """Full duplex listening with advanced AEC"""
    print("[Buddy V2] 🎤 Full Duplex Listening with Advanced AEC...")
    
    from audio.full_duplex_aec import full_duplex_aec
    
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                       channels=1, blocksize=blocksize, dtype='int16') as stream:
        
        # Quick baseline
        baseline_samples = []
        for _ in range(3):
            frame, _ = stream.read(blocksize)
            audio = np.frombuffer(frame.tobytes(), dtype=np.int16)
            audio_16k = downsample_audio(audio, MIC_SAMPLE_RATE, SAMPLE_RATE)
            volume = np.abs(audio_16k).mean()
            baseline_samples.append(volume)
        
        baseline = np.mean(baseline_samples) if baseline_samples else 200
        speech_threshold = max(baseline * 4.0, 800)  # Higher threshold for full duplex
        
        print(f"[FullDuplex] 👂 Ready (threshold: {speech_threshold:.0f})")
        
        audio_buffer = []
        start_time = time.time()
        silence_frames = 0
        has_speech = False
        speech_frames = 0
        required_speech_frames = 8  # More frames needed in full duplex
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= 8.0:
                break
            
            if has_speech and elapsed > 1.5 and silence_frames > 60:
                break
            
            try:
                frame, _ = stream.read(blocksize)
                audio = np.frombuffer(frame.tobytes(), dtype=np.int16)
                audio_16k = downsample_audio(audio, MIC_SAMPLE_RATE, SAMPLE_RATE)
                
                # Process with full duplex AEC
                for i in range(0, len(audio_16k), 160):
                    chunk = audio_16k[i:i+160]
                    if len(chunk) < 160:
                        continue
                    
                    # Apply advanced AEC
                    processed_chunk = full_duplex_aec.process_microphone_input(chunk)
                    audio_buffer.append(processed_chunk)
                    
                    # Volume detection
                    volume = np.abs(processed_chunk).mean()
                    peak_volume = np.max(np.abs(processed_chunk))
                    
                    volume_ok = volume > speech_threshold
                    peak_ok = peak_volume > speech_threshold * 1.5
                    
                    if volume_ok and peak_ok:
                        speech_frames += 1
                        silence_frames = 0
                        
                        if speech_frames >= required_speech_frames:
                            if not has_speech:
                                print(f"\n[FullDuplex] 🎤 USER speech detected!")
                                has_speech = True
                            print("🔴", end="", flush=True)
                        else:
                            print("🟡", end="", flush=True)
                    else:
                        speech_frames = max(0, speech_frames - 1)
                        silence_frames += 1
                        if has_speech and silence_frames % 30 == 0:
                            print("⏸️", end="", flush=True)
                    
            except Exception as e:
                if DEBUG:
                    print(f"\n[FullDuplex] Recording error: {e}")
                break
        
        # Process results
        if audio_buffer and len(audio_buffer) > 20:
            audio_np = np.concatenate(audio_buffer, axis=0).astype(np.int16)
            duration = len(audio_np) / SAMPLE_RATE
            volume = np.abs(audio_np).mean()
            
            if has_speech and volume > baseline * 3:
                aec_stats = full_duplex_aec.get_stats()
                print(f"\n[FullDuplex] ✅ Captured: {duration:.1f}s (AEC: {aec_stats['echo_cancellations']} cancellations)")
                return audio_np
            else:
                print(f"\n[FullDuplex] ⚠️ Low quality audio")
                return None
        else:
            print("\n[FullDuplex] ❌ No sufficient audio captured")
            return None

def half_duplex_vad_listen():
    """Half duplex listening (current approach)"""
    print("[Buddy V2] 🎤 Half Duplex Listening...")
    
    # Wait for Buddy to finish
    while buddy_talking.is_set():
        time.sleep(0.1)
    
    time.sleep(0.5)  # Safety buffer
    
    # Use existing AEC approach but with voice fingerprinting
    from audio.full_duplex_aec import full_duplex_aec
    
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                       channels=1, blocksize=blocksize, dtype='int16') as stream:
        
        baseline_samples = []
        for _ in range(3):
            frame, _ = stream.read(blocksize)
            audio = np.frombuffer(frame.tobytes(), dtype=np.int16)
            audio_16k = downsample_audio(audio, MIC_SAMPLE_RATE, SAMPLE_RATE)
            volume = np.abs(audio_16k).mean()
            baseline_samples.append(volume)
        
        baseline = np.mean(baseline_samples) if baseline_samples else 200
        speech_threshold = max(baseline * 3.0, 600)
        
        print(f"[HalfDuplex] 👂 Ready (threshold: {speech_threshold:.0f})")
        
        audio_buffer = []
        start_time = time.time()
        silence_frames = 0
        has_speech = False
        speech_frames = 0
        required_speech_frames = 5
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= 8.0:
                break
            
            if has_speech and elapsed > 1.0 and silence_frames > 40:
                break
            
            try:
                frame, _ = stream.read(blocksize)
                audio = np.frombuffer(frame.tobytes(), dtype=np.int16)
                audio_16k = downsample_audio(audio, MIC_SAMPLE_RATE, SAMPLE_RATE)
                
                # Process with voice fingerprinting
                for i in range(0, len(audio_16k), 160):
                    chunk = audio_16k[i:i+160]
                    if len(chunk) < 160:
                        continue
                    
                    # Apply AEC with voice fingerprinting
                    processed_chunk = full_duplex_aec.process_microphone_input(chunk)
                    audio_buffer.append(processed_chunk)
                    
                    volume = np.abs(processed_chunk).mean()
                    peak_volume = np.max(np.abs(processed_chunk))
                    
                    volume_ok = volume > speech_threshold
                    peak_ok = peak_volume > speech_threshold * 1.3
                    
                    if volume_ok and peak_ok:
                        speech_frames += 1
                        silence_frames = 0
                        
                        if speech_frames >= required_speech_frames:
                            if not has_speech:
                                print(f"\n[HalfDuplex] 🎤 USER speech detected!")
                                has_speech = True
                            print("🔴", end="", flush=True)
                        else:
                            print("🟡", end="", flush=True)
                    else:
                        speech_frames = max(0, speech_frames - 1)
                        silence_frames += 1
                        if has_speech and silence_frames % 20 == 0:
                            print("⏸️", end="", flush=True)
                    
            except Exception as e:
                if DEBUG:
                    print(f"\n[HalfDuplex] Recording error: {e}")
                break
        
        if audio_buffer and len(audio_buffer) > 15:
            audio_np = np.concatenate(audio_buffer, axis=0).astype(np.int16)
            duration = len(audio_np) / SAMPLE_RATE
            volume = np.abs(audio_np).mean()
            
            if has_speech and volume > baseline * 2:
                print(f"\n[HalfDuplex] ✅ Captured: {duration:.1f}s")
                return audio_np
            else:
                print(f"\n[HalfDuplex] ⚠️ Low quality audio")
                return None
        else:
            print("\n[HalfDuplex] ❌ No sufficient audio captured")
            return None

def aec_training_listen(description, timeout=8):
    """Training listen with voice fingerprinting"""
    while buddy_talking.is_set():
        time.sleep(0.1)
    
    time.sleep(1.0)
    
    return half_duplex_vad_listen()  # Use half duplex for training