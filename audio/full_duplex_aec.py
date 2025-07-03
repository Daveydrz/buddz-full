# audio/full_duplex_aec.py - Advanced AEC system (FIXED array broadcasting)
import numpy as np
import threading
import time
from scipy.signal import resample_poly
from pyaec import PyAec
from audio.voice_fingerprint import is_buddy_speaking, add_buddy_sample
from config import DEBUG, SAMPLE_RATE, FULL_DUPLEX_MODE, AEC_AGGRESSIVE_MODE

# Multi-stage AEC system
class FullDuplexAEC:
    def __init__(self):
        self.aec_primary = PyAec(frame_size=160, sample_rate=16000)
        self.aec_secondary = PyAec(frame_size=160, sample_rate=16000)
        
        # Reference buffers - FIXED SIZES
        self.ref_buffer_primary = np.zeros(32000, dtype=np.int16)    # 2 seconds at 16kHz
        self.ref_buffer_secondary = np.zeros(16000, dtype=np.int16)  # 1 second at 16kHz
        
        # Voice activity detection
        self.buddy_speaking_frames = 0
        self.voice_activity_threshold = 5
        
        # Locks
        self.ref_lock = threading.Lock()
        
        # Statistics
        self.echo_cancellations = 0
        self.voice_rejections = 0
        
        print("[FullDuplexAEC] ✅ Advanced AEC system initialized")
    
    def update_reference(self, pcm_data, sample_rate=16000):
        """Update AEC reference with Buddy's speech - FIXED"""
        try:
            # Convert to 16kHz if needed
            if sample_rate != 16000:
                pcm_float = pcm_data.astype(np.float32) / 32768.0
                pcm_16k = resample_poly(pcm_float, 16000, sample_rate)
                pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                pcm_16k = pcm_data.copy()
            
            # Add to voice fingerprint learning
            add_buddy_sample(pcm_16k)
            
            # Update reference buffers - FIXED ARRAY HANDLING
            with self.ref_lock:
                # Handle different array sizes properly
                pcm_len = len(pcm_16k)
                
                # Primary buffer update
                if pcm_len <= len(self.ref_buffer_primary):
                    # Shift buffer left and add new data
                    self.ref_buffer_primary = np.roll(self.ref_buffer_primary, -pcm_len)
                    self.ref_buffer_primary[-pcm_len:] = pcm_16k
                else:
                    # Data is larger than buffer - take the end
                    self.ref_buffer_primary[:] = pcm_16k[-len(self.ref_buffer_primary):]
                
                # Secondary buffer update
                if pcm_len <= len(self.ref_buffer_secondary):
                    self.ref_buffer_secondary = np.roll(self.ref_buffer_secondary, -pcm_len)
                    self.ref_buffer_secondary[-pcm_len:] = pcm_16k
                else:
                    # Take the most recent part that fits
                    self.ref_buffer_secondary[:] = pcm_16k[-len(self.ref_buffer_secondary):]
            
            if DEBUG:
                print(f"[FullDuplexAEC] 📡 Reference updated: {len(pcm_16k)} samples")
                
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplexAEC] Reference update error: {e}")
    
    def process_microphone_input(self, mic_chunk):
        """Process microphone input with multi-stage AEC"""
        try:
            # Stage 1: Voice fingerprint rejection
            if len(mic_chunk) >= 8000:  # 0.5 seconds
                if is_buddy_speaking(mic_chunk):
                    self.voice_rejections += 1
                    if DEBUG:
                        print(f"[FullDuplexAEC] 🚫 VOICE FINGERPRINT REJECTION")
                    # Return very quiet audio instead of silence
                    return (mic_chunk * 0.1).astype(np.int16)
            
            # Stage 2: Traditional AEC processing
            if len(mic_chunk) != 160:
                if len(mic_chunk) < 160:
                    mic_chunk = np.pad(mic_chunk, (0, 160 - len(mic_chunk)))
                else:
                    mic_chunk = mic_chunk[:160]
            
            # Convert to float32
            mic_float = mic_chunk.astype(np.float32) / 32768.0
            
            # Get reference frames
            with self.ref_lock:
                ref_primary = self.ref_buffer_primary[-160:].astype(np.float32) / 32768.0
                ref_secondary = self.ref_buffer_secondary[-160:].astype(np.float32) / 32768.0
            
            # Check reference strength
            ref_primary_rms = np.sqrt(np.mean(ref_primary ** 2))
            ref_secondary_rms = np.sqrt(np.mean(ref_secondary ** 2))
            mic_rms = np.sqrt(np.mean(mic_float ** 2))
            
            # Stage 3: Adaptive AEC processing
            if ref_primary_rms > 0.01 or ref_secondary_rms > 0.01:
                # Choose best reference
                if ref_secondary_rms > ref_primary_rms:
                    active_ref = ref_secondary
                    active_aec = self.aec_secondary
                else:
                    active_ref = ref_primary
                    active_aec = self.aec_primary
                
                try:
                    # Apply AEC
                    active_aec.set_ref(active_ref.tolist())
                    output = active_aec.process_with_ref(mic_float.tolist())
                    
                    if output and len(output) >= 160:
                        output_np = np.array(output[:160], dtype=np.float32)
                        output_rms = np.sqrt(np.mean(output_np ** 2))
                        
                        # Stage 4: Quality check
                        if AEC_AGGRESSIVE_MODE:
                            # Aggressive mode - always use AEC result
                            result = output_np
                            self.echo_cancellations += 1
                        else:
                            # Conservative mode - only use if significantly better
                            improvement_ratio = output_rms / (mic_rms + 1e-8)
                            if improvement_ratio < 0.8:  # 20% improvement
                                result = output_np
                                self.echo_cancellations += 1
                            else:
                                result = mic_float
                    else:
                        result = mic_float
                        
                except Exception as aec_err:
                    if DEBUG:
                        print(f"[FullDuplexAEC] AEC processing error: {aec_err}")
                    result = mic_float
            else:
                # No reference audio - pass through
                result = mic_float
            
            # Convert back to int16
            return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplexAEC] Processing error: {e}")
            return mic_chunk
    
    def get_stats(self):
        """Get AEC statistics"""
        return {
            "echo_cancellations": self.echo_cancellations,
            "voice_rejections": self.voice_rejections,
            "mode": "FULL_DUPLEX" if FULL_DUPLEX_MODE else "AGGRESSIVE_AEC"
        }

# Global AEC instance
full_duplex_aec = FullDuplexAEC()