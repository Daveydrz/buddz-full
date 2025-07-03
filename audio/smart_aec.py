# audio/smart_aec.py - Smart AEC that doesn't filter human speech
import numpy as np
from collections import deque
import time
from config import *

class SmartAEC:
    def __init__(self):
        self.reference_buffer = deque(maxlen=8000)  # 500ms reference
        self.adaptation_buffer = deque(maxlen=16000)  # 1 second adaptation
        self.echo_profile = None
        self.adaptation_rate = AEC_ADAPTATION_RATE
        self.suppression_factor = AEC_SUPPRESSION_FACTOR
        self.voice_detector = VoiceProtector()
        self.stats = {
            "echo_cancellations": 0,
            "voice_protections": 0,
            "adaptations": 0
        }
        print("[SmartAEC] ✅ Smart AEC initialized (VOICE PROTECTIVE)")
    
    def update_reference(self, reference_audio):
        """Update reference signal (Buddy's speech)"""
        try:
            if len(reference_audio) > 0:
                # Downsample if needed
                if len(reference_audio) > 8000:
                    reference_audio = reference_audio[::2][:8000]
                
                self.reference_buffer.extend(reference_audio)
                self.adaptation_buffer.extend(reference_audio)
                
                # Adapt echo profile gradually
                self._adapt_echo_profile()
                
        except Exception as e:
            if DEBUG:
                print(f"[SmartAEC] Reference update error: {e}")
    
    def process_microphone_input(self, mic_audio):
        """Process microphone input with smart echo cancellation"""
        try:
            if not AEC_ENABLED:
                return mic_audio
            
            # Check if this might be human speech
            if self.voice_detector.is_human_speech(mic_audio):
                self.stats["voice_protections"] += 1
                if DEBUG:
                    print(f"[SmartAEC] 👤 Human speech detected - minimal processing")
                return self._minimal_process(mic_audio)
            
            # Check if we have recent reference audio (Buddy speaking)
            if len(self.reference_buffer) < 1600:  # No recent reference
                return mic_audio
            
            # Apply conservative echo cancellation
            processed = self._conservative_echo_cancellation(mic_audio)
            
            if processed is not None:
                self.stats["echo_cancellations"] += 1
                return processed
            else:
                return mic_audio
                
        except Exception as e:
            if DEBUG:
                print(f"[SmartAEC] Processing error: {e}")
            return mic_audio
    
    def _minimal_process(self, audio):
        """Minimal processing for human speech"""
        try:
            # Only apply very gentle noise reduction
            if len(audio) > 160:
                # Calculate noise floor
                sorted_audio = np.sort(np.abs(audio))
                noise_floor = np.mean(sorted_audio[:len(sorted_audio)//4])
                
                # Very gentle noise gate
                noise_gate = noise_floor * 0.1
                mask = np.abs(audio) > noise_gate
                
                # Apply minimal noise reduction
                return np.where(mask, audio, audio * 0.8)
            
            return audio
            
        except Exception as e:
            return audio
    
    def _conservative_echo_cancellation(self, mic_audio):
        """Conservative echo cancellation that preserves speech"""
        try:
            if len(mic_audio) != len(self.reference_buffer):
                return None
            
            reference = np.array(list(self.reference_buffer))
            
            # Calculate correlation
            correlation = np.corrcoef(mic_audio, reference)[0, 1]
            
            # Only cancel if high correlation (likely echo)
            if correlation > 0.7:
                # Apply gentle suppression
                suppressed = mic_audio - (reference * self.suppression_factor)
                
                # Preserve speech characteristics
                volume_ratio = np.abs(mic_audio).mean() / (np.abs(reference).mean() + 1e-10)
                
                if volume_ratio > 0.3:  # Likely has human speech
                    # Mix original and suppressed
                    return mic_audio * 0.7 + suppressed * 0.3
                else:
                    return suppressed
            
            return mic_audio
            
        except Exception as e:
            return None
    
    def _adapt_echo_profile(self):
        """Adapt echo profile gradually"""
        try:
            if len(self.adaptation_buffer) >= 16000:
                # Simple adaptation - just update reference characteristics
                recent_audio = np.array(list(self.adaptation_buffer))
                
                # Update echo profile
                self.echo_profile = {
                    "volume": np.abs(recent_audio).mean(),
                    "peak": np.max(np.abs(recent_audio)),
                    "spectral_centroid": self._calculate_spectral_centroid(recent_audio)
                }
                
                self.stats["adaptations"] += 1
                
        except Exception as e:
            if DEBUG:
                print(f"[SmartAEC] Adaptation error: {e}")
    
    def _calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid for echo characterization"""
        try:
            # Simple spectral centroid approximation
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
            
            centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
            return centroid
            
        except Exception as e:
            return 1000  # Default value
    
    def get_stats(self):
        """Get AEC statistics"""
        return self.stats.copy()

class VoiceProtector:
    def __init__(self):
        self.human_voice_freq_range = (80, 255)  # Hz - Human fundamental frequency
        self.speech_pattern_buffer = deque(maxlen=10)
        
    def is_human_speech(self, audio):
        """Detect if audio contains human speech patterns"""
        try:
            if len(audio) < 160:
                return False
            
            # Volume check
            volume = np.abs(audio).mean()
            if volume < 100:  # Too quiet
                return False
            
            # Frequency analysis
            has_human_freqs = self._has_human_frequencies(audio)
            
            # Pattern analysis
            has_speech_pattern = self._has_speech_pattern(audio)
            
            # Dynamic range check
            peak = np.max(np.abs(audio))
            dynamic_range = peak / (volume + 1e-10)
            has_good_dynamics = 2.0 < dynamic_range < 20.0
            
            # Combined decision
            is_speech = (has_human_freqs and has_speech_pattern and has_good_dynamics)
            
            self.speech_pattern_buffer.append(is_speech)
            
            # Require consistent detection
            if len(self.speech_pattern_buffer) >= 3:
                return sum(self.speech_pattern_buffer) >= 2
            
            return is_speech
            
        except Exception as e:
            return False
    
    def _has_human_frequencies(self, audio):
        """Check for human voice frequencies"""
        try:
            # Simple frequency analysis
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
            
            # Check for energy in human voice range
            human_mask = (freqs >= self.human_voice_freq_range[0]) & (freqs <= self.human_voice_freq_range[1])
            human_energy = np.sum(magnitude[human_mask])
            total_energy = np.sum(magnitude)
            
            return (human_energy / (total_energy + 1e-10)) > 0.1
            
        except Exception as e:
            return False
    
    def _has_speech_pattern(self, audio):
        """Check for speech-like patterns"""
        try:
            # Look for amplitude variations typical of speech
            if len(audio) < 320:  # 20ms
                return False
            
            # Split into small windows
            window_size = 80  # 5ms windows
            windows = [audio[i:i+window_size] for i in range(0, len(audio)-window_size, window_size)]
            
            if len(windows) < 3:
                return False
            
            # Calculate volume variations
            volumes = [np.abs(w).mean() for w in windows]
            
            # Check for speech-like modulation
            volume_std = np.std(volumes)
            volume_mean = np.mean(volumes)
            
            modulation_ratio = volume_std / (volume_mean + 1e-10)
            
            # Speech typically has moderate modulation
            return 0.2 < modulation_ratio < 2.0
            
        except Exception as e:
            return False

# Global instance
smart_aec = SmartAEC()