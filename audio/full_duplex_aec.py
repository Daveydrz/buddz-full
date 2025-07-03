# audio/full_duplex_aec.py - Complete Smart Conservative AEC (FULL FEATURED)
import numpy as np
import threading
import time
from scipy.signal import resample_poly
from pyaec import PyAec
from audio.voice_fingerprint import is_buddy_speaking, add_buddy_sample

# Safe config loading with all required constants
try:
    from config import (DEBUG, SAMPLE_RATE, FULL_DUPLEX_MODE, AEC_AGGRESSIVE_MODE,
                       BUDDY_VOICE_THRESHOLD, VOICE_SIMILARITY_BUFFER, VOICE_LEARNING_PATIENCE)
except ImportError:
    DEBUG = True
    SAMPLE_RATE = 16000
    FULL_DUPLEX_MODE = True
    AEC_AGGRESSIVE_MODE = False
    BUDDY_VOICE_THRESHOLD = 0.90
    VOICE_SIMILARITY_BUFFER = 5
    VOICE_LEARNING_PATIENCE = 10
    print("[FullDuplexAEC] ⚠️ Using default AEC settings")

# Multi-stage SMART CONSERVATIVE AEC system
class FullDuplexAEC:
    def __init__(self):
        self.aec_primary = PyAec(frame_size=160, sample_rate=16000)
        self.aec_secondary = PyAec(frame_size=160, sample_rate=16000)
        
        # Reference buffers - FIXED SIZES
        self.ref_buffer_primary = np.zeros(32000, dtype=np.int16)    # 2 seconds at 16kHz
        self.ref_buffer_secondary = np.zeros(16000, dtype=np.int16)  # 1 second at 16kHz
        
        # SMART FILTERING - Track voice activity patterns
        self.buddy_speaking_frames = 0
        self.voice_activity_threshold = 8  # Increased from 5 - more conservative
        self.human_speech_detector = HumanSpeechDetector()
        
        # ADVANCED FILTERING CONTROLS
        self.adaptive_suppression = AdaptiveSuppressionController()
        self.voice_quality_monitor = VoiceQualityMonitor()
        
        # Locks
        self.ref_lock = threading.Lock()
        
        # Statistics
        self.echo_cancellations = 0
        self.voice_rejections = 0
        self.human_speech_protected = 0
        self.passthrough_count = 0
        self.quality_improvements = 0
        self.adaptive_adjustments = 0
        
        print("[FullDuplexAEC] ✅ Complete Smart Conservative AEC system initialized")
    
    def update_reference(self, pcm_data, sample_rate=16000):
        """Update AEC reference with Buddy's speech - ENHANCED"""
        try:
            # Convert to 16kHz if needed
            if sample_rate != 16000:
                pcm_float = pcm_data.astype(np.float32) / 32768.0
                pcm_16k = resample_poly(pcm_float, 16000, sample_rate)
                pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                pcm_16k = pcm_data.copy()
            
            # Add to voice fingerprint learning (gradually)
            if len(pcm_16k) >= 16000:  # Only learn from substantial samples
                add_buddy_sample(pcm_16k)
            
            # Update voice quality monitoring
            self.voice_quality_monitor.update_reference_quality(pcm_16k)
            
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
        """Process microphone input with COMPLETE SMART CONSERVATIVE AEC"""
        try:
            # Stage 1: HUMAN SPEECH PROTECTION - Priority check
            human_speech_confidence = self.human_speech_detector.analyze_speech_confidence(mic_chunk)
            
            if human_speech_confidence > 0.7:  # High confidence human speech
                self.human_speech_protected += 1
                if DEBUG:
                    print(f"[FullDuplexAEC] 👤 HUMAN SPEECH PROTECTED (conf: {human_speech_confidence:.3f})")
                return self._minimal_processing(mic_chunk)
            
            # Stage 2: ADVANCED voice fingerprint analysis
            if len(mic_chunk) >= 12800:  # 0.8 seconds - longer requirement
                buddy_confidence = self._get_comprehensive_buddy_confidence(mic_chunk)
                
                if buddy_confidence > 0.92:  # Very high confidence required
                    self.voice_rejections += 1
                    if DEBUG:
                        print(f"[FullDuplexAEC] 🚫 BUDDY VOICE REJECTED (confidence: {buddy_confidence:.3f})")
                    # Return very quiet audio instead of silence
                    return (mic_chunk * 0.05).astype(np.int16)
                    
                elif buddy_confidence > 0.85:
                    if DEBUG:
                        print(f"[FullDuplexAEC] ⚠️ Possible Buddy voice ({buddy_confidence:.3f}) - applying gentle processing")
                    return self._gentle_processing(mic_chunk)
                
                elif buddy_confidence > 0.7:
                    if DEBUG:
                        print(f"[FullDuplexAEC] 🤔 Uncertain voice ({buddy_confidence:.3f}) - conservative processing")
                    return self._conservative_processing(mic_chunk)
            
            # Stage 3: Adaptive AEC processing based on context
            return self._adaptive_aec_processing(mic_chunk)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplexAEC] Processing error: {e}")
            return mic_chunk
    
    def _minimal_processing(self, mic_chunk):
        """Minimal processing for confirmed human speech"""
        try:
            # Only apply very gentle noise reduction
            if len(mic_chunk) > 320:  # 20ms
                # Calculate noise floor from quietest parts
                abs_chunk = np.abs(mic_chunk)
                sorted_abs = np.sort(abs_chunk)
                noise_floor = np.mean(sorted_abs[:len(sorted_abs)//5])  # Bottom 20%
                
                # Very gentle noise gate
                noise_gate = noise_floor * 0.2
                mask = abs_chunk > noise_gate
                
                # Apply minimal noise reduction
                result = np.where(mask, mic_chunk, mic_chunk * 0.9)
                return result.astype(np.int16)
            
            return mic_chunk
            
        except Exception as e:
            return mic_chunk
    
    def _gentle_processing(self, mic_chunk):
        """Gentle processing for possible Buddy voice"""
        try:
            # Apply moderate suppression with quality preservation
            suppression_factor = self.adaptive_suppression.get_suppression_factor(mic_chunk)
            return (mic_chunk * suppression_factor).astype(np.int16)
        except Exception as e:
            return mic_chunk
    
    def _conservative_processing(self, mic_chunk):
        """Conservative processing for uncertain voice"""
        try:
            # Apply light suppression
            return (mic_chunk * 0.7).astype(np.int16)
        except Exception as e:
            return mic_chunk
    
    def _get_comprehensive_buddy_confidence(self, mic_chunk):
        """Get comprehensive confidence that this is Buddy's voice"""
        try:
            confidences = []
            
            # Multiple sample analysis
            if len(mic_chunk) >= 16000:  # 1 second
                # Split into overlapping segments
                segments = [
                    mic_chunk[:8000],
                    mic_chunk[4000:12000],
                    mic_chunk[8000:16000]
                ]
                
                for segment in segments:
                    if len(segment) >= 8000:
                        conf = 0.33 if is_buddy_speaking(segment) else 0.0
                        confidences.append(conf)
            
            elif len(mic_chunk) >= 8000:
                conf = 0.5 if is_buddy_speaking(mic_chunk) else 0.0
                confidences.append(conf)
            
            # Voice quality analysis
            quality_score = self.voice_quality_monitor.analyze_voice_quality(mic_chunk)
            if quality_score > 0.8:  # High quality suggests TTS
                confidences.append(0.2)
            
            return sum(confidences)
                
        except Exception as e:
            return 0.0
    
    def _adaptive_aec_processing(self, mic_chunk):
        """Adaptive AEC processing with comprehensive quality preservation"""
        try:
            # Normalize chunk size
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
            
            # Adaptive threshold based on recent performance
            ref_threshold = self.adaptive_suppression.get_reference_threshold()
            
            # Only apply AEC if there's significant reference audio
            if ref_primary_rms > ref_threshold or ref_secondary_rms > ref_threshold:
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
                        
                        # ADAPTIVE quality check
                        improvement_threshold = self.adaptive_suppression.get_improvement_threshold()
                        improvement_ratio = output_rms / (mic_rms + 1e-8)
                        
                        if improvement_ratio < improvement_threshold:
                            result = output_np
                            self.echo_cancellations += 1
                            self.adaptive_suppression.record_success()
                        else:
                            result = mic_float
                            self.passthrough_count += 1
                            self.adaptive_suppression.record_passthrough()
                    else:
                        result = mic_float
                        self.passthrough_count += 1
                        
                except Exception as aec_err:
                    if DEBUG:
                        print(f"[FullDuplexAEC] AEC processing error: {aec_err}")
                    result = mic_float
                    self.passthrough_count += 1
            else:
                # No significant reference audio - pass through
                result = mic_float
                self.passthrough_count += 1
            
            # Convert back to int16
            return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplexAEC] Adaptive AEC processing error: {e}")
            return mic_chunk
    
    def get_stats(self):
        """Get comprehensive AEC statistics"""
        return {
            "echo_cancellations": self.echo_cancellations,
            "voice_rejections": self.voice_rejections,
            "human_speech_protected": self.human_speech_protected,
            "passthrough_count": self.passthrough_count,
            "quality_improvements": self.quality_improvements,
            "adaptive_adjustments": self.adaptive_adjustments,
            "mode": "COMPLETE_SMART_CONSERVATIVE" if not AEC_AGGRESSIVE_MODE else "ENHANCED_CONSERVATIVE_AGGRESSIVE",
            "adaptive_suppression_stats": self.adaptive_suppression.get_stats(),
            "voice_quality_stats": self.voice_quality_monitor.get_stats()
        }

class HumanSpeechDetector:
    """Advanced human speech pattern detection"""
    
    def __init__(self):
        self.human_freq_range = (85, 300)  # Hz - Human fundamental frequency range
        self.speech_indicators = []
        self.confidence_history = []
        
    def analyze_speech_confidence(self, audio_chunk):
        """Analyze confidence that this is human speech"""
        try:
            if len(audio_chunk) < 320:  # Need at least 20ms
                return 0.0
            
            confidence_factors = []
            
            # Volume analysis
            volume = np.abs(audio_chunk).mean()
            if volume < 80:
                return 0.0
            
            volume_confidence = min(1.0, volume / 500.0)
            confidence_factors.append(volume_confidence * 0.3)
            
            # Dynamic range analysis
            peak = np.max(np.abs(audio_chunk))
            dynamic_range = peak / (volume + 1e-10)
            
            if 1.5 <= dynamic_range <= 25:
                range_confidence = 1.0 - abs(dynamic_range - 8) / 8  # Optimal around 8
                confidence_factors.append(range_confidence * 0.3)
            
            # Frequency analysis
            if len(audio_chunk) >= 1024:
                freq_confidence = self._analyze_frequency_confidence(audio_chunk)
                confidence_factors.append(freq_confidence * 0.2)
            
            # Temporal pattern analysis
            temporal_confidence = self._analyze_temporal_confidence(audio_chunk)
            confidence_factors.append(temporal_confidence * 0.2)
            
            # Combine all factors
            total_confidence = sum(confidence_factors)
            
            # Store for trend analysis
            self.confidence_history.append(total_confidence)
            if len(self.confidence_history) > 10:
                self.confidence_history.pop(0)
            
            return total_confidence
            
        except Exception as e:
            if DEBUG:
                print(f"[HumanSpeechDetector] Analysis error: {e}")
            return 0.0
    
    def is_human_speech(self, audio_chunk):
        """Simple boolean check for human speech"""
        return self.analyze_speech_confidence(audio_chunk) > 0.7
    
    def _analyze_frequency_confidence(self, audio_chunk):
        """Analyze frequency content confidence"""
        try:
            fft = np.fft.rfft(audio_chunk)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/16000)
            
            # Check for energy in human speech range
            speech_mask = (freqs >= self.human_freq_range[0]) & (freqs <= self.human_freq_range[1])
            speech_energy = np.sum(magnitude[speech_mask])
            total_energy = np.sum(magnitude)
            
            speech_ratio = speech_energy / (total_energy + 1e-10)
            
            # Check for harmonic structure
            harmonic_confidence = self._detect_harmonic_structure(magnitude, freqs)
            
            return min(1.0, speech_ratio * 3 + harmonic_confidence * 0.5)
            
        except Exception as e:
            return 0.0
    
    def _detect_harmonic_structure(self, magnitude, freqs):
        """Detect harmonic structure typical of human speech"""
        try:
            if len(magnitude) < 50:
                return 0.0
            
            # Find peaks
            peaks = []
            for i in range(2, len(magnitude)-2):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if magnitude[i] > np.mean(magnitude) * 2:
                        peaks.append((i, magnitude[i]))
            
            if len(peaks) < 2:
                return 0.0
            
            # Check for harmonic relationships
            harmonic_pairs = 0
            for i, (peak1_idx, peak1_mag) in enumerate(peaks):
                for j, (peak2_idx, peak2_mag) in enumerate(peaks[i+1:], i+1):
                    freq1 = freqs[peak1_idx]
                    freq2 = freqs[peak2_idx]
                    
                    # Check if freq2 is approximately a harmonic of freq1
                    if freq1 > 0:
                        harmonic_ratio = freq2 / freq1
                        if 1.8 <= harmonic_ratio <= 2.2 or 2.8 <= harmonic_ratio <= 3.2:
                            harmonic_pairs += 1
            
            return min(1.0, harmonic_pairs / 2.0)
            
        except Exception as e:
            return 0.0
    
    def _analyze_temporal_confidence(self, audio_chunk):
        """Analyze temporal pattern confidence"""
        try:
            if len(audio_chunk) < 640:
                return 0.0
            
            # Split into windows
            window_size = 160
            windows = []
            
            for i in range(0, len(audio_chunk) - window_size, window_size):
                window = audio_chunk[i:i+window_size]
                windows.append(np.abs(window).mean())
            
            if len(windows) < 3:
                return 0.0
            
            # Analyze variations
            variations = []
            for i in range(1, len(windows)):
                variation = abs(windows[i] - windows[i-1])
                variations.append(variation)
            
            avg_variation = np.mean(variations)
            modulation_ratio = avg_variation / (np.mean(windows) + 1e-10)
            
            # Check for natural speech patterns
            if 0.1 <= modulation_ratio <= 1.5:
                modulation_confidence = 1.0 - abs(modulation_ratio - 0.5) / 0.5
            else:
                modulation_confidence = 0.0
            
            # Check for natural pauses
            quiet_threshold = np.mean(windows) * 0.3
            quiet_windows = sum(1 for w in windows if w < quiet_threshold)
            pause_ratio = quiet_windows / len(windows)
            
            if 0.1 <= pause_ratio <= 0.4:
                pause_confidence = 1.0 - abs(pause_ratio - 0.25) / 0.25
            else:
                pause_confidence = 0.0
            
            return (modulation_confidence + pause_confidence) / 2.0
            
        except Exception as e:
            return 0.0

class AdaptiveSuppressionController:
    """Controls adaptive suppression based on performance"""
    
    def __init__(self):
        self.success_rate = 0.5
        self.recent_successes = []
        self.reference_threshold = 0.02
        self.improvement_threshold = 0.6
        self.adjustment_history = []
        
    def get_suppression_factor(self, audio_chunk):
        """Get adaptive suppression factor"""
        try:
            # Base suppression
            base_factor = 0.3
            
            # Adjust based on recent performance
            if self.success_rate > 0.8:
                # High success rate - can be more aggressive
                return base_factor * 0.8
            elif self.success_rate < 0.3:
                # Low success rate - be more conservative
                return base_factor * 1.5
            
            return base_factor
            
        except Exception as e:
            return 0.3
    
    def get_reference_threshold(self):
        """Get adaptive reference threshold"""
        return self.reference_threshold
    
    def get_improvement_threshold(self):
        """Get adaptive improvement threshold"""
        return self.improvement_threshold
    
    def record_success(self):
        """Record successful AEC operation"""
        self.recent_successes.append(1)
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        self._update_success_rate()
    
    def record_passthrough(self):
        """Record passthrough operation"""
        self.recent_successes.append(0)
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        self._update_success_rate()
    
    def _update_success_rate(self):
        """Update success rate and adjust thresholds"""
        if len(self.recent_successes) >= 5:
            self.success_rate = np.mean(self.recent_successes)
            
            # Adjust thresholds based on success rate
            if self.success_rate > 0.8:
                self.reference_threshold = min(0.03, self.reference_threshold * 1.1)
                self.improvement_threshold = min(0.8, self.improvement_threshold * 1.05)
            elif self.success_rate < 0.3:
                self.reference_threshold = max(0.01, self.reference_threshold * 0.9)
                self.improvement_threshold = max(0.4, self.improvement_threshold * 0.95)
    
    def get_stats(self):
        """Get suppression controller stats"""
        return {
            "success_rate": self.success_rate,
            "reference_threshold": self.reference_threshold,
            "improvement_threshold": self.improvement_threshold,
            "recent_operations": len(self.recent_successes)
        }

class VoiceQualityMonitor:
    """Monitors voice quality for TTS detection"""
    
    def __init__(self):
        self.reference_qualities = []
        self.microphone_qualities = []
        
    def update_reference_quality(self, audio_data):
        """Update reference audio quality metrics"""
        try:
            quality = self._calculate_quality_score(audio_data)
            self.reference_qualities.append(quality)
            if len(self.reference_qualities) > 50:
                self.reference_qualities.pop(0)
        except Exception as e:
            pass
    
    def analyze_voice_quality(self, audio_data):
        """Analyze voice quality (higher = more TTS-like)"""
        try:
            quality = self._calculate_quality_score(audio_data)
            self.microphone_qualities.append(quality)
            if len(self.microphone_qualities) > 50:
                self.microphone_qualities.pop(0)
            
            return quality
            
        except Exception as e:
            return 0.5
    
    def _calculate_quality_score(self, audio_data):
        """Calculate audio quality score"""
        try:
            if len(audio_data) < 1600:
                return 0.5
            
            # Consistency measure
            segments = [audio_data[i:i+800] for i in range(0, len(audio_data)-800, 800)]
            segment_volumes = [np.abs(seg).mean() for seg in segments if len(seg) == 800]
            
            if len(segment_volumes) < 2:
                return 0.5
            
            volume_std = np.std(segment_volumes)
            volume_mean = np.mean(segment_volumes)
            
            consistency = 1.0 - min(1.0, volume_std / (volume_mean + 1e-10))
            
            # Spectral smoothness
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            
            # Calculate spectral smoothness
            spectral_diff = np.diff(magnitude)
            spectral_smoothness = 1.0 - min(1.0, np.std(spectral_diff) / (np.mean(magnitude) + 1e-10))
            
            # Combine metrics (TTS typically has higher consistency and smoothness)
            quality_score = (consistency * 0.6 + spectral_smoothness * 0.4)
            
            return quality_score
            
        except Exception as e:
            return 0.5
    
    def get_stats(self):
        """Get voice quality statistics"""
        return {
            "reference_avg_quality": np.mean(self.reference_qualities) if self.reference_qualities else 0.0,
            "microphone_avg_quality": np.mean(self.microphone_qualities) if self.microphone_qualities else 0.0,
            "reference_samples": len(self.reference_qualities),
            "microphone_samples": len(self.microphone_qualities)
        }

# Global AEC instance
full_duplex_aec = FullDuplexAEC()