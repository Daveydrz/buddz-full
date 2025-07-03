# audio/full_duplex_manager.py - Complete Full Duplex Manager (UPDATED WITH CORRECT CONFIG)
import threading
import time
import queue
import numpy as np
from collections import deque
from config import *

class FullDuplexManager:
    def __init__(self):
        # Audio streams
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=50)
        self.processed_queue = queue.Queue(maxsize=50)
        
        # State management
        self.listening = False
        self.processing = False
        self.buddy_interrupted = False
        
        # ENHANCED audio buffers for perfect speech capture
        self.mic_buffer = deque(maxlen=4800)      # 300ms at 16kHz
        self.speech_buffer = deque(maxlen=240000) # 15 seconds at 16kHz
        self.pre_speech_buffer = deque(maxlen=16000) # 1 second pre-speech context
        
        # OPTIMIZED Voice Activity Detection (using config values)
        self.vad_threshold = VAD_THRESHOLD            # From config: 200
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_frames = MIN_SPEECH_FRAMES    # From config: 4
        self.max_silence_frames = MAX_SILENCE_FRAMES  # From config: 150
        
        # Interrupt detection (using config values)
        self.interrupt_threshold = INTERRUPT_THRESHOLD  # From config: 1500
        self.interrupt_frames = 0
        self.min_interrupt_frames = 8
        
        # ENHANCED noise filtering
        self.noise_baseline = 150
        self.noise_samples = deque(maxlen=200)
        self.noise_calibrated = False
        self.adaptive_threshold_history = deque(maxlen=50)
        
        # SPEECH QUALITY IMPROVEMENTS
        self.speech_quality_buffer = deque(maxlen=32)
        self.consecutive_good_frames = 0
        self.last_good_speech_time = 0
        
        # Statistics
        self.interrupts_detected = 0
        self.speeches_processed = 0
        self.buddy_speeches_rejected = 0
        self.false_detections = 0
        self.enhanced_captures = 0
        self.quality_improvements = 0
        
        # Threading
        self.running = False
        self.threads = []
        
        print("[FullDuplex] 🚀 Full Duplex Manager initialized (UPDATED WITH CONFIG)")
    
    def start(self):
        """Start full duplex communication"""
        if self.running:
            return
        
        self.running = True
        self.listening = True
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._audio_input_worker, daemon=True),
            threading.Thread(target=self._vad_processor, daemon=True),
            threading.Thread(target=self._speech_processor, daemon=True),
            threading.Thread(target=self._interrupt_detector, daemon=True),
            threading.Thread(target=self._noise_tracker, daemon=True),
            threading.Thread(target=self._pre_speech_tracker, daemon=True),
            threading.Thread(target=self._quality_monitor, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        print("[FullDuplex] ✅ Full duplex communication started")
    
    def stop(self):
        """Stop full duplex communication"""
        self.running = False
        self.listening = False
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
        
        print("[FullDuplex] 🛑 Full duplex communication stopped")
    
    def add_audio_input(self, audio_chunk):
        """Add microphone audio input"""
        if not self.listening:
            return
        
        try:
            # Import here to avoid circular imports
            from audio.full_duplex_aec import full_duplex_aec
            
            # Apply smart AEC
            processed_chunk = full_duplex_aec.process_microphone_input(audio_chunk)
            
            # Add to input queue for processing
            if not self.input_queue.full():
                self.input_queue.put(processed_chunk)
            
            # Add to real-time buffer
            self.mic_buffer.extend(processed_chunk)
            
            # ALWAYS add to pre-speech buffer for context
            self.pre_speech_buffer.extend(processed_chunk)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Audio input error: {e}")
    
    def notify_buddy_speaking(self, audio_data):
        """Notify that Buddy is speaking (for AEC reference)"""
        try:
            from audio.full_duplex_aec import full_duplex_aec
            full_duplex_aec.update_reference(audio_data)
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Notify buddy speaking error: {e}")
    
    def _pre_speech_tracker(self):
        """Enhanced pre-speech tracking"""
        while self.running:
            try:
                # The buffer is automatically maintained in add_audio_input
                # Monitor quality here
                if len(self.pre_speech_buffer) >= 1600:  # 100ms
                    recent_chunk = np.array(list(self.pre_speech_buffer)[-1600:])
                    quality = self._assess_audio_quality(recent_chunk)
                    self.speech_quality_buffer.append(quality)
                
                time.sleep(0.05)  # 50ms cycle
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Pre-speech tracker error: {e}")
    
    def _quality_monitor(self):
        """Monitor speech quality in real-time"""
        while self.running:
            try:
                if len(self.speech_quality_buffer) >= 10:
                    avg_quality = np.mean(list(self.speech_quality_buffer)[-10:])
                    
                    # Adjust thresholds based on quality
                    if avg_quality > 0.7:  # Good quality
                        self.consecutive_good_frames += 1
                        if self.consecutive_good_frames >= 20:
                            # Lower threshold for good quality audio
                            self.vad_threshold = max(VAD_THRESHOLD * 0.5, self.vad_threshold * 0.98)
                    else:
                        self.consecutive_good_frames = 0
                        # Slightly raise threshold for poor quality
                        self.vad_threshold = min(VAD_THRESHOLD * 2, self.vad_threshold * 1.01)
                
                time.sleep(0.2)  # 200ms cycle
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Quality monitor error: {e}")
    
    def _assess_audio_quality(self, audio_chunk):
        """Assess audio quality (0.0 to 1.0)"""
        try:
            if len(audio_chunk) == 0:
                return 0.0
            
            # Calculate various quality metrics
            volume = np.abs(audio_chunk).mean()
            peak = np.max(np.abs(audio_chunk))
            
            # Signal-to-noise ratio estimate
            sorted_abs = np.sort(np.abs(audio_chunk))
            noise_level = np.mean(sorted_abs[:len(sorted_abs)//4])  # Bottom 25%
            signal_level = np.mean(sorted_abs[3*len(sorted_abs)//4:])  # Top 25%
            
            snr = signal_level / (noise_level + 1e-10)
            
            # Dynamic range
            dynamic_range = peak / (volume + 1e-10)
            
            # Combine metrics
            quality = min(1.0, (snr * 0.4 + dynamic_range * 0.3 + min(volume/1000, 1.0) * 0.3))
            
            return quality
            
        except Exception as e:
            return 0.5  # Default quality
    
    def _noise_tracker(self):
        """Enhanced noise tracking with calibration"""
        calibration_samples = 0
        
        while self.running:
            try:
                if len(self.mic_buffer) >= 160:
                    chunk = np.array(list(self.mic_buffer)[-160:])
                    volume = np.abs(chunk).mean()
                    
                    # Only track during silence (no processing)
                    if not self.processing:
                        self.noise_samples.append(volume)
                        calibration_samples += 1
                        
                        # Initial calibration period
                        if not self.noise_calibrated and calibration_samples >= 50:
                            self.noise_baseline = np.percentile(self.noise_samples, 75)
                            self.noise_calibrated = True
                            print(f"[FullDuplex] 🎯 Noise calibrated: {self.noise_baseline:.1f}")
                        
                        # Update baseline every 10 seconds
                        elif len(self.noise_samples) >= 100:
                            old_baseline = self.noise_baseline
                            self.noise_baseline = np.percentile(self.noise_samples, 75)
                            
                            # Gradual adaptation
                            self.noise_baseline = old_baseline * 0.8 + self.noise_baseline * 0.2
                            
                            if DEBUG and len(self.noise_samples) % 100 == 0:
                                print(f"[FullDuplex] 📊 Noise baseline updated: {self.noise_baseline:.1f}")
                
                time.sleep(0.1)
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Noise tracker error: {e}")
    
    def _audio_input_worker(self):
        """Process audio input with smart filtering"""
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.input_queue.get(timeout=0.1)
                
                # Add to speech buffer for VAD (AEC already applied)
                self.speech_buffer.extend(audio_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Input worker error: {e}")
    
   def _vad_processor(self):
    """SIMPLIFIED and MORE SENSITIVE Voice Activity Detection processor"""
    consecutive_speech_frames = 0
    
    while self.running:
        try:
            if len(self.speech_buffer) < 160:
                time.sleep(0.01)
                continue
            
            # Get chunk for VAD
            chunk = np.array(list(self.speech_buffer)[-160:])
            volume = np.abs(chunk).mean()
            peak = np.max(np.abs(chunk))
            
            # MUCH SIMPLER threshold calculation
            if self.noise_calibrated:
                adaptive_threshold = max(self.vad_threshold, self.noise_baseline * 1.5)
            else:
                adaptive_threshold = self.vad_threshold
            
            # SIMPLIFIED speech detection - much more sensitive
            volume_ok = volume > adaptive_threshold
            peak_ok = peak > adaptive_threshold * 1.1
            
            # Much more lenient decision
            is_speech = volume_ok or peak_ok
            
            # ENHANCED DEBUG - show every 50 frames to avoid spam
            if DEBUG and (volume > 50 or peak > 50) and consecutive_speech_frames % 50 == 0:
                print(f"[VAD DEBUG] vol:{volume:.1f}, peak:{peak:.1f}, thresh:{adaptive_threshold:.1f}, speech:{is_speech}")
                print(f"[VAD DEBUG] speech_frames:{self.speech_frames}, silence_frames:{self.silence_frames}, processing:{self.processing}")
            
            if is_speech:
                self.speech_frames += 1
                consecutive_speech_frames += 1
                self.silence_frames = 0
                
                # Check for speech start
                if self.speech_frames >= self.min_speech_frames and not self.processing:
                    print(f"\n[FullDuplex] 🎤 USER SPEECH DETECTED! (vol:{volume:.1f}, thresh:{adaptive_threshold:.1f})")
                    print(f"[FullDuplex] Starting speech capture...")
                    self._start_speech_capture_enhanced()
                    
            else:
                consecutive_speech_frames = 0
                self.speech_frames = max(0, self.speech_frames - 1)
                if self.processing:
                    self.silence_frames += 1
                    
                    # Check for speech end
                    if self.silence_frames >= self.max_silence_frames:
                        print(f"\n[FullDuplex] ⏸️ Speech ended (silence frames: {self.silence_frames})")
                        self._end_speech_capture_enhanced()
            
            time.sleep(0.01)  # 10ms processing cycle
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] VAD processor error: {e}")
    
    def _interrupt_detector(self):
        """Interrupt detection"""
        while self.running:
            try:
                if not INTERRUPT_DETECTION:
                    time.sleep(0.1)
                    continue
                
                if len(self.mic_buffer) < 160:
                    time.sleep(0.01)
                    continue
                
                # Check for interrupt-level speech
                chunk = np.array(list(self.mic_buffer)[-160:])
                volume = np.abs(chunk).mean()
                
                # Use interrupt threshold from config
                if volume > self.interrupt_threshold:
                    self.interrupt_frames += 1
                    
                    if self.interrupt_frames >= self.min_interrupt_frames:
                        # Check if Buddy is currently speaking
                        try:
                            from audio.output import current_audio_playback
                            if current_audio_playback and current_audio_playback.is_playing():
                                print(f"\n[FullDuplex] ⚡ USER INTERRUPT DETECTED! (vol:{volume:.1f})")
                                self._handle_interrupt()
                                self.interrupt_frames = 0  # Reset
                        except:
                            pass
                else:
                    self.interrupt_frames = max(0, self.interrupt_frames - 1)
                
                time.sleep(0.01)
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Interrupt detector error: {e}")
    
    def _start_speech_capture_enhanced(self):
        """Start capturing user speech with maximum context"""
        self.processing = True
        self.speech_start_time = time.time()
        self.last_good_speech_time = time.time()
        
        # Add pre-speech context for better transcription (using config values)
        pre_context_frames = int(SPEECH_PADDING_START * SAMPLE_RATE)  # 0.8s from config
        if len(self.pre_speech_buffer) >= pre_context_frames:
            pre_context = list(self.pre_speech_buffer)[-pre_context_frames:]
            
            # Clear and rebuild speech buffer with context
            self.speech_buffer.clear()
            self.speech_buffer.extend(pre_context)
            
            print(f"[FullDuplex] 📝 Added {len(pre_context)} pre-speech samples")
        
        print("🔴", end="", flush=True)
    
    def _end_speech_capture_enhanced(self):
        """End speech capture with quality enhancement"""
        if not self.processing:
            return
        
        self.processing = False
        
        # Wait for complete word endings (using config values)
        time.sleep(SPEECH_PADDING_END)  # 0.8s from config
        
        # Get context for Whisper (using config values)
        total_context_frames = int(WHISPER_CONTEXT_PADDING * SAMPLE_RATE)  # 3.0s from config
        
        if len(self.speech_buffer) > total_context_frames:
            audio_data = np.array(list(self.speech_buffer)[-total_context_frames:])
        else:
            audio_data = np.array(list(self.speech_buffer))
        
        # Quality validation
        duration = len(audio_data) / SAMPLE_RATE
        volume = np.abs(audio_data).mean()
        
        # Apply gentle enhancement
        if len(audio_data) > SAMPLE_RATE:
            audio_data = self._enhance_audio_quality(audio_data)
            self.quality_improvements += 1
        
        # Lenient acceptance criteria (using config values)
        min_duration = MIN_SPEECH_DURATION  # 0.2s from config
        volume_threshold = self.noise_baseline * 1.5
        
        if duration >= min_duration and volume > volume_threshold:
            print(f"\n[FullDuplex] ✅ Captured speech: {duration:.1f}s (vol:{volume:.1f})")
            
            # Add to processing queue
            self.processed_queue.put(audio_data)
            self.speeches_processed += 1
            self.enhanced_captures += 1
        else:
            print(f"\n[FullDuplex] ❌ Speech rejected: {duration:.1f}s (vol:{volume:.1f})")
            self.false_detections += 1
        
        # Keep context for next capture
        keep_frames = int(0.8 * SAMPLE_RATE)
        if len(self.speech_buffer) > keep_frames:
            kept_audio = list(self.speech_buffer)[-keep_frames:]
            self.speech_buffer.clear()
            self.speech_buffer.extend(kept_audio)
        else:
            self.speech_buffer.clear()
    
    def _enhance_audio_quality(self, audio_data):
        """Enhance audio quality for better transcription"""
        try:
            # Convert to float for processing
            audio_float = audio_data.astype(np.float32)
            
            # Calculate noise profile
            abs_audio = np.abs(audio_float)
            noise_level = np.percentile(abs_audio, 20)
            
            # Gentle noise gate
            noise_gate = noise_level * 0.4
            mask = abs_audio > noise_gate
            
            # Apply noise reduction
            audio_float = np.where(mask, audio_float, audio_float * 0.3)
            
            # Gentle normalization
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                target_max = 18000  # Leave headroom
                audio_float = audio_float * (target_max / max_val)
            
            return audio_float.astype(np.int16)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Audio enhancement error: {e}")
            return audio_data
    
    def _handle_interrupt(self):
        """Handle user interruption"""
        self.interrupts_detected += 1
        self.buddy_interrupted = True
        
        # Stop current Buddy speech
        try:
            from audio.output import current_audio_playback
            if current_audio_playback:
                try:
                    current_audio_playback.stop()
                    print(f"[FullDuplex] 🛑 Buddy speech interrupted")
                except:
                    pass
        except:
            pass
        
        # Start capturing interrupt speech
        self._start_speech_capture_enhanced()
    
    def _speech_processor(self):
        """Process captured speech"""
        while self.running:
            try:
                # Get speech to process
                audio_data = self.processed_queue.get(timeout=0.5)
                
                # Transcribe in background
                def transcribe_and_handle():
                    try:
                        from ai.speech import transcribe_audio
                        
                        if DEBUG:
                            duration = len(audio_data) / SAMPLE_RATE
                            volume = np.abs(audio_data).mean()
                            quality = self._assess_audio_quality(audio_data)
                            print(f"[FullDuplex] 🎙️ Sending to Whisper: {duration:.1f}s, vol:{volume:.1f}, quality:{quality:.2f}")
                        
                        text = transcribe_audio(audio_data)
                        if text and len(text.strip()) > 0:
                            print(f"[FullDuplex] 📝 Transcribed: '{text}'")
                            self._handle_transcribed_text(text, audio_data)
                        else:
                            print(f"[FullDuplex] ❌ Empty transcription")
                    except Exception as e:
                        if DEBUG:
                            print(f"[FullDuplex] Transcription error: {e}")
                
                threading.Thread(target=transcribe_and_handle, daemon=True).start()
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Speech processor error: {e}")
    
    def _handle_transcribed_text(self, text, audio_data):
        """Handle transcribed text"""
        # Store for main conversation handler
        self.last_transcription = (text, audio_data)
    
    def get_next_speech(self, timeout=0.1):
        """Get next processed speech (non-blocking)"""
        try:
            if hasattr(self, 'last_transcription'):
                result = self.last_transcription
                delattr(self, 'last_transcription')
                return result
        except:
            pass
        return None
    
    def get_stats(self):
        """Get enhanced statistics"""
        try:
            from audio.voice_fingerprint import get_buddy_voice_status
            from audio.full_duplex_aec import full_duplex_aec
            
            buddy_status = get_buddy_voice_status()
            aec_stats = full_duplex_aec.get_stats()
            
            return {
                "running": self.running,
                "listening": self.listening,
                "processing": self.processing,
                "interrupts_detected": self.interrupts_detected,
                "speeches_processed": self.speeches_processed,
                "enhanced_captures": self.enhanced_captures,
                "quality_improvements": self.quality_improvements,
                "buddy_rejections": self.buddy_speeches_rejected,
                "false_detections": self.false_detections,
                "noise_calibrated": self.noise_calibrated,
                "buddy_voice_learned": buddy_status.get("learned", False),
                "aec_stats": aec_stats,
                "noise_baseline": self.noise_baseline,
                "adaptive_vad_threshold": self.vad_threshold,
                "config_values": {
                    "VAD_THRESHOLD": VAD_THRESHOLD,
                    "MIN_SPEECH_FRAMES": MIN_SPEECH_FRAMES,
                    "MAX_SILENCE_FRAMES": MAX_SILENCE_FRAMES,
                    "INTERRUPT_THRESHOLD": INTERRUPT_THRESHOLD,
                    "MIN_SPEECH_DURATION": MIN_SPEECH_DURATION,
                    "SPEECH_PADDING_START": SPEECH_PADDING_START,
                    "SPEECH_PADDING_END": SPEECH_PADDING_END,
                    "WHISPER_CONTEXT_PADDING": WHISPER_CONTEXT_PADDING
                },
                "buffer_sizes": {
                    "input": self.input_queue.qsize(),
                    "processed": self.processed_queue.qsize(),
                    "mic_buffer": len(self.mic_buffer),
                    "speech_buffer": len(self.speech_buffer),
                    "pre_speech_buffer": len(self.pre_speech_buffer)
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "running": self.running,
                "listening": self.listening
            }

# Create and export the global instance
try:
    full_duplex_manager = FullDuplexManager()
    print("[FullDuplex] ✅ Global full duplex manager created (UPDATED)")
except Exception as e:
    print(f"[FullDuplex] ❌ Error creating manager: {e}")
    full_duplex_manager = None