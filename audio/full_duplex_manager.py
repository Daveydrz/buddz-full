# audio/full_duplex_manager.py - Full Duplex Manager (IMPROVED SPEECH CAPTURE)
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
        
        # IMPROVED audio buffers for better speech capture
        self.mic_buffer = deque(maxlen=3200)      # 200ms at 16kHz
        self.speech_buffer = deque(maxlen=160000) # 10 seconds at 16kHz
        self.pre_speech_buffer = deque(maxlen=8000) # 500ms pre-speech context
        
        # IMPROVED Voice Activity Detection
        self.vad_threshold = VAD_THRESHOLD
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_frames = MIN_SPEECH_FRAMES
        self.max_silence_frames = MAX_SILENCE_FRAMES
        
        # IMPROVED Interrupt detection
        self.interrupt_threshold = INTERRUPT_THRESHOLD
        self.interrupt_frames = 0
        self.min_interrupt_frames = 8
        
        # NOISE FILTERING
        self.noise_baseline = 200
        self.noise_samples = deque(maxlen=100)
        
        # SPEECH CAPTURE IMPROVEMENTS
        self.speech_start_buffer = []  # Store audio before speech detection
        self.speech_end_padding = []   # Store audio after speech ends
        self.capturing_pre_speech = False
        
        # Statistics
        self.interrupts_detected = 0
        self.speeches_processed = 0
        self.buddy_speeches_rejected = 0
        self.false_detections = 0
        self.improved_captures = 0
        
        # Threading
        self.running = False
        self.threads = []
        
        print("[FullDuplex] 🚀 Full Duplex Manager initialized (IMPROVED SPEECH CAPTURE)")
    
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
            threading.Thread(target=self._pre_speech_tracker, daemon=True)  # NEW
        ]
        
        for thread in self.threads:
            thread.start()
        
        print("[FullDuplex] ✅ Full duplex communication started (IMPROVED)")
    
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
            
            # Apply full duplex AEC
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
        """NEW: Track audio before speech detection"""
        while self.running:
            try:
                # This just maintains the pre_speech_buffer
                # The buffer is automatically maintained in add_audio_input
                time.sleep(0.1)
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Pre-speech tracker error: {e}")
    
    def _noise_tracker(self):
        """Track background noise level"""
        while self.running:
            try:
                if len(self.mic_buffer) >= 160:
                    chunk = np.array(list(self.mic_buffer)[-160:])
                    volume = np.abs(chunk).mean()
                    
                    # Only track during silence (no processing)
                    if not self.processing:
                        self.noise_samples.append(volume)
                        
                        # Update baseline every 10 seconds
                        if len(self.noise_samples) >= 100:
                            self.noise_baseline = np.percentile(self.noise_samples, 70)  # 70th percentile
                            if DEBUG and len(self.noise_samples) % 100 == 0:
                                print(f"[FullDuplex] 📊 Noise baseline updated: {self.noise_baseline:.1f}")
                
                time.sleep(0.1)
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Noise tracker error: {e}")
    
    def _audio_input_worker(self):
        """Process audio input continuously"""
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.input_queue.get(timeout=0.1)
                
                # Check if this is Buddy's voice (voice fingerprinting)
                if len(audio_chunk) >= 1600:  # 100ms
                    try:
                        from audio.voice_fingerprint import is_buddy_speaking
                        recent_audio = np.array(list(self.mic_buffer)[-1600:])
                        if is_buddy_speaking(recent_audio):
                            self.buddy_speeches_rejected += 1
                            if DEBUG:
                                print(f"[FullDuplex] 🚫 Buddy voice rejected")
                            continue
                    except Exception as e:
                        if DEBUG:
                            print(f"[FullDuplex] Voice fingerprint error: {e}")
                
                # Add to speech buffer for VAD
                self.speech_buffer.extend(audio_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Input worker error: {e}")
    
    def _vad_processor(self):
        """IMPROVED Voice Activity Detection processor"""
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
                
                # IMPROVED threshold based on noise baseline
                adaptive_threshold = max(self.vad_threshold, self.noise_baseline * 2.5)  # Lower multiplier
                
                # MORE SENSITIVE Voice activity detection
                volume_ok = volume > adaptive_threshold
                peak_ok = peak > adaptive_threshold * 1.2  # Lower peak requirement
                is_speech = volume_ok and peak_ok
                
                if is_speech:
                    self.speech_frames += 1
                    consecutive_speech_frames += 1
                    self.silence_frames = 0
                    
                    # Check for speech start - MORE RESPONSIVE
                    if self.speech_frames >= self.min_speech_frames and not self.processing:
                        print(f"\n[FullDuplex] 🎤 USER SPEECH DETECTED! (vol:{volume:.1f}, thresh:{adaptive_threshold:.1f})")
                        self._start_speech_capture_improved()
                        
                else:
                    consecutive_speech_frames = 0
                    self.speech_frames = max(0, self.speech_frames - 1)
                    if self.processing:
                        self.silence_frames += 1
                        
                        # Check for speech end - MORE PATIENT
                        if self.silence_frames >= self.max_silence_frames:
                            print(f"\n[FullDuplex] ⏸️ Speech ended (silence frames: {self.silence_frames})")
                            self._end_speech_capture_improved()
                
                time.sleep(0.01)  # 10ms processing cycle
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] VAD processor error: {e}")
    
    def _interrupt_detector(self):
        """IMPROVED interrupt detection"""
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
                
                # Use higher threshold for interrupts
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
    
    def _start_speech_capture_improved(self):
        """IMPROVED: Start capturing user speech with context"""
        self.processing = True
        self.speech_start_time = time.time()
        
        # Add pre-speech context for better transcription
        pre_context_frames = int(SPEECH_PADDING_START * SAMPLE_RATE)
        if len(self.pre_speech_buffer) >= pre_context_frames:
            pre_context = list(self.pre_speech_buffer)[-pre_context_frames:]
            self.speech_buffer.extend(pre_context)
            print(f"[FullDuplex] 📝 Added {len(pre_context)} pre-speech samples")
        
        print("🔴", end="", flush=True)
    
    def _end_speech_capture_improved(self):
        """IMPROVED: End speech capture with padding and better validation"""
        if not self.processing:
            return
        
        self.processing = False
        
        # Add post-speech padding for complete words
        post_padding_frames = int(SPEECH_PADDING_END * SAMPLE_RATE)
        padding_collected = 0
        
        # Wait a bit to collect ending padding
        time.sleep(SPEECH_PADDING_END)
        
        # Get captured audio from buffer with extra context
        total_context_frames = int(WHISPER_CONTEXT_PADDING * SAMPLE_RATE)
        
        if len(self.speech_buffer) > total_context_frames:
            # Take more context for better transcription
            audio_data = np.array(list(self.speech_buffer)[-total_context_frames:])
        else:
            audio_data = np.array(list(self.speech_buffer))
        
        # IMPROVED duration and quality checks
        min_duration = MIN_SPEECH_DURATION
        duration = len(audio_data) / SAMPLE_RATE
        volume = np.abs(audio_data).mean()
        
        # More lenient quality check
        if duration >= min_duration and volume > self.noise_baseline * 1.5:  # Lower threshold
            print(f"\n[FullDuplex] ✅ Captured speech: {duration:.1f}s (vol:{volume:.1f}) [IMPROVED]")
            
            # Add to processing queue
            self.processed_queue.put(audio_data)
            self.speeches_processed += 1
            self.improved_captures += 1
        else:
            print(f"\n[FullDuplex] ❌ Speech rejected: {duration:.1f}s (vol:{volume:.1f}, min:{min_duration}s)")
            self.false_detections += 1
        
        # Clear speech buffer but keep some context
        keep_frames = int(0.5 * SAMPLE_RATE)  # Keep 500ms context
        if len(self.speech_buffer) > keep_frames:
            # Keep the last 500ms as context for next speech
            kept_audio = list(self.speech_buffer)[-keep_frames:]
            self.speech_buffer.clear()
            self.speech_buffer.extend(kept_audio)
        else:
            self.speech_buffer.clear()
    
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
        
        # Start capturing interrupt speech with improved method
        self._start_speech_capture_improved()
    
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
                        
                        # Save audio for debugging if needed
                        if DEBUG:
                            duration = len(audio_data) / SAMPLE_RATE
                            volume = np.abs(audio_data).mean()
                            print(f"[FullDuplex] 🎙️ Sending to Whisper: {duration:.1f}s, vol:{volume:.1f}")
                        
                        text = transcribe_audio(audio_data)
                        if text and len(text.strip()) > 0:
                            print(f"[FullDuplex] 📝 Transcribed: '{text}' [IMPROVED CAPTURE]")
                            # Handle the transcribed text
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
        """Get full duplex statistics"""
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
                "improved_captures": self.improved_captures,
                "buddy_rejections": self.buddy_speeches_rejected,
                "false_detections": self.false_detections,
                "buddy_voice_learned": buddy_status["learned"],
                "aec_cancellations": aec_stats.get("echo_cancellations", 0),
                "noise_baseline": self.noise_baseline,
                "vad_threshold": self.vad_threshold,
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
    print("[FullDuplex] ✅ Global full duplex manager created (IMPROVED SPEECH CAPTURE)")
except Exception as e:
    print(f"[FullDuplex] ❌ Error creating manager: {e}")
    full_duplex_manager = None