# audio/aec.py - SMART Acoustic Echo Cancellation
import numpy as np
import threading
import time
from scipy.signal import resample_poly
from pyaec import PyAec
from config import DEBUG, SAMPLE_RATE

# AEC (Acoustic Echo Cancellation) setup
aec_instance = PyAec(frame_size=160, sample_rate=16000)
ref_audio_buffer = np.zeros(16000 * 2, dtype=np.int16)  # 2 seconds buffer
ref_audio_lock = threading.Lock()
aec_active = threading.Event()
last_buddy_speech_time = None

def update_aec_reference(pcm_data, sample_rate=16000):
    """Update AEC reference with Buddy's speech - SMART VERSION"""
    global ref_audio_buffer, last_buddy_speech_time
    
    try:
        from audio.output import buddy_talking
        
        if not buddy_talking.is_set():
            return
        
        # Mark when Buddy last spoke
        last_buddy_speech_time = time.time()
        aec_active.set()
        
        # Convert to 16kHz if needed
        if sample_rate != 16000:
            pcm_float = pcm_data.astype(np.float32) / 32768.0
            pcm_16k = resample_poly(pcm_float, 16000, sample_rate)
            pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm_16k = pcm_data.copy()
        
        if DEBUG:
            print(f"[AEC] 📡 BUDDY SPEAKING - Recording reference: {len(pcm_16k)} samples")
        
        # Update reference buffer in 160-sample chunks
        chunk_size = 160
        chunks_processed = 0
        
        for i in range(0, len(pcm_16k), chunk_size):
            if not buddy_talking.is_set():
                break
                
            frame = pcm_16k[i:i+chunk_size]
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)))
            
            with ref_audio_lock:
                ref_audio_buffer = np.roll(ref_audio_buffer, -chunk_size)
                ref_audio_buffer[-chunk_size:] = frame
            
            chunks_processed += 1
            time.sleep(0.01)  # 10ms per chunk
        
        if DEBUG:
            print(f"[AEC] ✅ Reference updated: {chunks_processed} chunks")
            
        # Keep AEC active for a SHORT time after speech ends
        time.sleep(0.3)  # Only 300ms
        aec_active.clear()
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Reference update error: {e}")
        aec_active.clear()

def apply_aec_to_microphone(mic_chunk):
    """Apply AEC to microphone input - SMART VERSION (only when actually needed)"""
    global ref_audio_buffer, last_buddy_speech_time
    
    try:
        from audio.output import buddy_talking
        
        # CRITICAL: Only apply AEC when actually needed
        buddy_is_talking_now = buddy_talking.is_set()
        
        # Check if Buddy spoke VERY recently (within 1 second)
        recently_talked = (last_buddy_speech_time and 
                          time.time() - last_buddy_speech_time < 1.0)
        
        # ONLY apply AEC if Buddy is currently talking OR just finished (within 1 second)
        if not (buddy_is_talking_now or recently_talked):
            # NO AEC - return original microphone input
            return mic_chunk
        
        if DEBUG and buddy_is_talking_now:
            print(f"[AEC] 🔇 APPLYING AEC - Buddy is talking")
        elif DEBUG and recently_talked:
            print(f"[AEC] 🔇 APPLYING AEC - Buddy just finished ({time.time() - last_buddy_speech_time:.1f}s ago)")
        
        # Ensure correct chunk size
        if len(mic_chunk) != 160:
            if len(mic_chunk) < 160:
                mic_chunk = np.pad(mic_chunk, (0, 160 - len(mic_chunk)))
            else:
                mic_chunk = mic_chunk[:160]
        
        # Convert to float32
        mic_float = mic_chunk.astype(np.float32) / 32768.0
        
        # Get reference frame
        with ref_audio_lock:
            ref_frame = ref_audio_buffer[-160:].astype(np.float32) / 32768.0
        
        # Check if we have valid reference audio
        ref_rms = np.sqrt(np.mean(ref_frame ** 2))
        mic_rms = np.sqrt(np.mean(mic_float ** 2))
        
        if ref_rms < 0.005:  # Reference too quiet
            if DEBUG:
                print(f"[AEC] ⚠️ Reference too quiet: {ref_rms:.6f} - using original")
            return mic_chunk
        
        # Apply AEC processing
        try:
            aec_instance.set_ref(ref_frame.tolist())
            output = aec_instance.process_with_ref(mic_float.tolist())
            
            if output and len(output) >= 160:
                output_np = np.array(output[:160], dtype=np.float32)
                
                # Check if AEC actually helped
                mic_norm = mic_float / (np.linalg.norm(mic_float) + 1e-8)
                ref_norm = ref_frame / (np.linalg.norm(ref_frame) + 1e-8)
                similarity_before = abs(np.dot(mic_norm, ref_norm))
                
                output_norm = output_np / (np.linalg.norm(output_np) + 1e-8)
                similarity_after = abs(np.dot(output_norm, ref_norm))
                
                # Only use AEC if it significantly reduces echo
                if similarity_after < similarity_before * 0.8:  # Must reduce by at least 20%
                    result = output_np
                    if DEBUG:
                        print(f"[AEC] ✅ Echo reduced: {similarity_before:.3f} → {similarity_after:.3f}")
                else:
                    # AEC didn't help much - use original
                    result = mic_float
                    if DEBUG:
                        print(f"[AEC] 🔄 AEC ineffective, using original")
            else:
                result = mic_float
                
        except Exception as aec_err:
            if DEBUG:
                print(f"[AEC] Processing error: {aec_err}")
            result = mic_float  # Use original on error
        
        # Convert back to int16
        return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Error: {e}")
        return mic_chunk  # Return original on error

def get_aec_status():
    """Get current AEC status for debugging"""
    try:
        from audio.output import buddy_talking
        
        with ref_audio_lock:
            ref_level = np.sqrt(np.mean((ref_audio_buffer[-160:].astype(np.float32) / 32768.0) ** 2))
        
        recently_talked = (last_buddy_speech_time and 
                          time.time() - last_buddy_speech_time < 1.0)
        
        return {
            "buddy_talking": buddy_talking.is_set(),
            "aec_active": aec_active.is_set(),
            "recently_talked": recently_talked,
            "last_speech": last_buddy_speech_time,
            "time_since_speech": time.time() - last_buddy_speech_time if last_buddy_speech_time else None,
            "ref_level": ref_level
        }
    except:
        return {"error": "Cannot get AEC status"}