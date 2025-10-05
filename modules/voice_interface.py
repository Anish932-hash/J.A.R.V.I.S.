"""
J.A.R.V.I.S. Voice Interface
Advanced voice recognition and text-to-speech system
"""

import os
import time
import threading
import queue
import wave
import audioop
import tempfile
from typing import Optional, Dict, Any, List, Callable
import logging

# Windows-specific imports for TTS and STT
try:
    import win32com.client
    import pythoncom
    import pyttsx3
    import speech_recognition as sr
    import pyaudio
except ImportError as e:
    print(f"Warning: Some voice modules not available: {e}")


class VoiceInterface:
    """
    Advanced voice interface for J.A.R.V.I.S.
    Handles speech recognition and text-to-speech
    """

    def __init__(self, jarvis_instance):
        """
        Initialize voice interface

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.VoiceInterface')

        # TTS (Text-to-Speech) setup
        self.tts_engine = None
        self.tts_initialized = False
        self.current_voice = None
        self.speech_rate = 200
        self.speech_volume = 0.8

        # STT (Speech-to-Text) setup
        self.recognizer = None
        self.microphone = None
        self.stt_initialized = False
        self.listening = False
        self.listen_thread = None

        # Voice settings
        self.wake_word = "jarvis"
        self.confidence_threshold = 0.7
        self.timeout = 5
        self.phrase_time_limit = 10

        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024

        # Voice activity detection
        self.vad_threshold = 300  # Voice activity detection threshold
        self.silence_duration = 1.0  # Seconds of silence before stopping

        # Queues for communication
        self.speech_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Callbacks
        self.speech_callbacks = []
        self.command_callbacks = []

        # Performance tracking
        self.stats = {
            "total_speech_commands": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "total_speech_output": 0,
            "average_confidence": 0.0
        }

    def initialize(self):
        """Initialize voice interface components"""
        try:
            self.logger.info("Initializing voice interface...")

            # Initialize TTS
            self._initialize_tts()

            # Initialize STT
            self._initialize_stt()

            # Start speech processing thread
            self._start_speech_processor()

            self.logger.info("Voice interface initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing voice interface: {e}")
            raise

    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            # Try pyttsx3 first (cross-platform)
            self.tts_engine = pyttsx3.init()

            # Configure voice properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a good English voice
                for voice in voices:
                    if 'english' in voice.languages or 'en' in str(voice.languages).lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break

            self.tts_engine.setProperty('rate', self.speech_rate)
            self.tts_engine.setProperty('volume', self.speech_volume)

            # Fallback to Windows SAPI if pyttsx3 fails
            if not self.tts_engine:
                pythoncom.CoInitialize()
                self.tts_engine = win32com.client.Dispatch("SAPI.SpVoice")

            self.tts_initialized = True
            self.logger.info("TTS engine initialized")

        except Exception as e:
            self.logger.error(f"Error initializing TTS: {e}")
            self.logger.info("TTS will be unavailable")

    def _initialize_stt(self):
        """Initialize speech-to-text engine"""
        try:
            self.recognizer = sr.Recognizer()

            # Configure recognizer
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8

            # Get microphone
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)

            # Adjust for ambient noise
            with self.microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)

            self.stt_initialized = True
            self.logger.info("STT engine initialized")

        except Exception as e:
            self.logger.error(f"Error initializing STT: {e}")
            self.logger.info("STT will be unavailable")

    def _start_speech_processor(self):
        """Start speech processing thread"""
        if not hasattr(self, '_speech_processor'):
            self._speech_processor = threading.Thread(
                target=self._speech_processing_loop,
                name="SpeechProcessor",
                daemon=True
            )
            self._speech_processor.start()

    def speak(self, text: str, priority: str = "normal", callback: Callable = None):
        """
        Speak text using TTS

        Args:
            text: Text to speak
            priority: Priority level (low, normal, high, critical)
            callback: Optional callback when speech completes
        """
        if not self.tts_initialized or not text:
            return

        try:
            # Add to speech queue
            speech_item = {
                "text": text,
                "priority": priority,
                "timestamp": time.time(),
                "callback": callback
            }

            self.speech_queue.put(speech_item)

            # Update stats
            self.stats["total_speech_output"] += 1

            self.logger.debug(f"Queued speech: {text[:50]}...")

        except Exception as e:
            self.logger.error(f"Error queuing speech: {e}")

    def listen(self, timeout: int = None, phrase_time_limit: int = None) -> Optional[str]:
        """
        Listen for speech input

        Args:
            timeout: Maximum time to listen in seconds
            phrase_time_limit: Maximum time for a single phrase

        Returns:
            Recognized text or None if no speech detected
        """
        if not self.stt_initialized:
            self.logger.warning("STT not initialized")
            return None

        timeout = timeout or self.timeout
        phrase_time_limit = phrase_time_limit or self.phrase_time_limit

        try:
            self.logger.info("Listening for speech...")

            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            # Recognize speech
            text = self._recognize_audio(audio)

            if text:
                self.logger.info(f"Recognized: {text}")
                self.stats["successful_recognitions"] += 1

                # Check for wake word
                if self.wake_word.lower() in text.lower():
                    # Remove wake word and get command
                    command = text.lower().replace(self.wake_word.lower(), "").strip()
                    if command:
                        self._process_voice_command(command, text)
                        return command

            return text

        except sr.WaitTimeoutError:
            self.logger.debug("Listening timeout")
        except sr.UnknownValueError:
            self.logger.debug("Could not understand audio")
            self.stats["failed_recognitions"] += 1
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            self.stats["failed_recognitions"] += 1
        except Exception as e:
            self.logger.error(f"Error listening for speech: {e}")
            self.stats["failed_recognitions"] += 1

        return None

    def _recognize_audio(self, audio) -> Optional[str]:
        """Recognize speech from audio data with enhanced providers"""
        try:
            # Try Azure Speech Services first (if available)
            if hasattr(self, '_azure_speech_key') and self._azure_speech_key:
                text = self._recognize_azure(audio)
                if text:
                    return text

            # Try Google Cloud Speech-to-Text (if available)
            if hasattr(self, '_google_speech_credentials') and self._google_speech_credentials:
                text = self._recognize_google_cloud(audio)
                if text:
                    return text

            # Try Google Speech Recognition (free tier)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass

            # Try Sphinx as final fallback
            try:
                text = self.recognizer.recognize_sphinx(audio)
                return text
            except:
                pass

        except Exception as e:
            self.logger.error(f"Error in enhanced speech recognition: {e}")

        return None

    def _recognize_azure(self, audio) -> Optional[str]:
        """Recognize speech using Azure Speech Services"""
        try:
            import azure.cognitiveservices.speech as speechsdk

            # Get Azure credentials from environment or config
            subscription_key = os.getenv("AZURE_SPEECH_KEY")
            region = os.getenv("AZURE_SPEECH_REGION", "eastus")

            if not subscription_key:
                return None

            # Configure speech recognition
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            speech_config.speech_recognition_language = "en-US"

            # Use audio data
            stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=stream)

            # Create speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Push audio data
            audio_data = audio.get_wav_data()
            stream.write(audio_data)
            stream.close()

            # Recognize speech
            result = speech_recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                self.logger.debug("Azure: No speech could be recognized")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                self.logger.debug(f"Azure: Speech recognition canceled: {cancellation_details.reason}")

            return None

        except ImportError:
            self.logger.debug("Azure Speech SDK not available")
            return None
        except Exception as e:
            self.logger.debug(f"Azure speech recognition error: {e}")
            return None

    def _recognize_google_cloud(self, audio) -> Optional[str]:
        """Recognize speech using Google Cloud Speech-to-Text"""
        try:
            from google.cloud import speech
            import io

            # Get credentials from environment
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                return None

            # Initialize client
            client = speech.SpeechClient()

            # Convert audio to proper format
            audio_content = audio.get_wav_data()

            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-US",
            )

            # Create audio object
            audio_obj = speech.RecognitionAudio(content=audio_content)

            # Perform recognition
            response = client.recognize(config=config, audio=audio_obj)

            # Extract results
            for result in response.results:
                if result.alternatives:
                    return result.alternatives[0].transcript

            return None

        except ImportError:
            self.logger.debug("Google Cloud Speech client not available")
            return None
        except Exception as e:
            self.logger.debug(f"Google Cloud speech recognition error: {e}")
            return None

    def start_continuous_listening(self):
        """Start continuous listening for voice commands"""
        if self.listening:
            return

        self.listening = True
        self.listen_thread = threading.Thread(
            target=self._continuous_listen_loop,
            name="ContinuousListener",
            daemon=True
        )
        self.listen_thread.start()
        self.logger.info("Continuous listening started")

    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.listening = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        self.logger.info("Continuous listening stopped")

    def _continuous_listen_loop(self):
        """Continuous listening loop"""
        while self.listening:
            try:
                command = self.listen(timeout=3)
                if command:
                    # Process the command
                    self.jarvis.execute_command(command, {"source": "voice"})

                time.sleep(0.1)  # Small delay between listening attempts

            except Exception as e:
                self.logger.error(f"Error in continuous listening: {e}")
                time.sleep(1)

    def _process_voice_command(self, command: str, original_text: str):
        """Process voice command"""
        try:
            # Update stats
            self.stats["total_speech_commands"] += 1

            # Calculate confidence (simple implementation)
            confidence = 0.8  # Would be more sophisticated in real implementation

            # Update average confidence
            total_confidence = self.stats["average_confidence"] * (self.stats["total_speech_commands"] - 1)
            self.stats["average_confidence"] = (total_confidence + confidence) / self.stats["total_speech_commands"]

            # Trigger callbacks
            for callback in self.command_callbacks:
                try:
                    callback(command, original_text, confidence)
                except Exception as e:
                    self.logger.error(f"Error in command callback: {e}")

            self.logger.info(f"Processed voice command: {command}")

        except Exception as e:
            self.logger.error(f"Error processing voice command: {e}")

    def _speech_processing_loop(self):
        """Process speech queue"""
        while True:
            try:
                # Get speech item from queue
                speech_item = self.speech_queue.get(timeout=1)

                if not speech_item:
                    continue

                # Speak the text
                self._speak_text(speech_item["text"])

                # Call callback if provided
                if speech_item["callback"]:
                    try:
                        speech_item["callback"]()
                    except Exception as e:
                        self.logger.error(f"Error in speech callback: {e}")

                self.speech_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in speech processing loop: {e}")
                time.sleep(1)

    def _speak_text(self, text: str):
        """Speak text using TTS engine"""
        try:
            if hasattr(self.tts_engine, 'say'):
                # pyttsx3 engine
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Windows SAPI
                self.tts_engine.Speak(text)

            self.logger.debug(f"Spoke: {text[:50]}...")

        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")

    def set_voice(self, voice_name: str = None):
        """Set TTS voice"""
        try:
            if not self.tts_initialized:
                return False

            if hasattr(self.tts_engine, 'setProperty'):
                # pyttsx3
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if voice_name.lower() in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            self.current_voice = voice.id
                            return True
            else:
                # Windows SAPI - would need different implementation
                pass

            return False

        except Exception as e:
            self.logger.error(f"Error setting voice: {e}")
            return False

    def set_speech_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        try:
            if not self.tts_initialized:
                return False

            self.speech_rate = max(50, min(400, rate))  # Clamp to reasonable range

            if hasattr(self.tts_engine, 'setProperty'):
                self.tts_engine.setProperty('rate', self.speech_rate)

            return True

        except Exception as e:
            self.logger.error(f"Error setting speech rate: {e}")
            return False

    def set_speech_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        try:
            if not self.tts_initialized:
                return False

            self.speech_volume = max(0.0, min(1.0, volume))

            if hasattr(self.tts_engine, 'setProperty'):
                self.tts_engine.setProperty('volume', self.speech_volume)

            return True

        except Exception as e:
            self.logger.error(f"Error setting speech volume: {e}")
            return False

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        try:
            if not self.tts_initialized:
                return []

            voices = []

            if hasattr(self.tts_engine, 'getProperty'):
                # pyttsx3
                engine_voices = self.tts_engine.getProperty('voices')
                for voice in engine_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "languages": voice.languages,
                        "gender": getattr(voice, 'gender', 'unknown'),
                        "age": getattr(voice, 'age', 'unknown')
                    })
            else:
                # Windows SAPI - would need different implementation
                pass

            return voices

        except Exception as e:
            self.logger.error(f"Error getting available voices: {e}")
            return []

    def test_speech(self) -> bool:
        """Test TTS functionality"""
        try:
            test_text = "J.A.R.V.I.S. voice interface test successful."
            self.speak(test_text)

            # Wait a moment for speech to complete
            time.sleep(2)

            return True

        except Exception as e:
            self.logger.error(f"Error testing speech: {e}")
            return False

    def test_listening(self) -> bool:
        """Test speech recognition functionality"""
        try:
            self.logger.info("Testing listening for 5 seconds...")
            result = self.listen(timeout=5)

            if result:
                self.logger.info(f"Successfully recognized: {result}")
                return True
            else:
                self.logger.info("No speech detected during test")
                return False

        except Exception as e:
            self.logger.error(f"Error testing listening: {e}")
            return False

    def add_speech_callback(self, callback: Callable):
        """Add speech callback"""
        self.speech_callbacks.append(callback)

    def add_command_callback(self, callback: Callable):
        """Add voice command callback"""
        self.command_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get voice interface statistics"""
        return {
            **self.stats,
            "tts_initialized": self.tts_initialized,
            "stt_initialized": self.stt_initialized,
            "listening": self.listening,
            "wake_word": self.wake_word,
            "speech_rate": self.speech_rate,
            "speech_volume": self.speech_volume,
            "confidence_threshold": self.confidence_threshold
        }

    def save_audio(self, audio_data, filename: str):
        """Save audio data to file"""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            filepath = os.path.join(temp_dir, filename)

            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.get_wav_data())

            return filepath

        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
            return None

    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        try:
            if hasattr(self.tts_engine, 'isBusy'):
                return self.tts_engine.isBusy()
            return False
        except:
            return False

    def stop_speaking(self):
        """Stop current speech"""
        try:
            if hasattr(self.tts_engine, 'stop'):
                self.tts_engine.stop()
        except Exception as e:
            self.logger.error(f"Error stopping speech: {e}")

    def set_wake_word(self, wake_word: str):
        """Set wake word for voice activation"""
        self.wake_word = wake_word.lower().strip()
        self.logger.info(f"Wake word set to: {self.wake_word}")

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for speech recognition"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def calibrate_microphone(self, duration: int = 3):
        """Calibrate microphone for ambient noise"""
        try:
            if not self.stt_initialized:
                return False

            self.logger.info(f"Calibrating microphone for {duration} seconds...")

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)

            self.logger.info("Microphone calibration complete")
            return True

        except Exception as e:
            self.logger.error(f"Error calibrating microphone: {e}")
            return False

    def shutdown(self):
        """Shutdown voice interface"""
        try:
            self.logger.info("Shutting down voice interface...")

            # Stop continuous listening
            self.stop_continuous_listening()

            # Stop speech processing
            if hasattr(self, '_speech_processor'):
                # Clear queue and stop thread
                while not self.speech_queue.empty():
                    try:
                        self.speech_queue.get_nowait()
                    except queue.Empty:
                        break

            # Close TTS engine
            if self.tts_engine:
                try:
                    if hasattr(self.tts_engine, 'stop'):
                        self.tts_engine.stop()
                except:
                    pass

            self.logger.info("Voice interface shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down voice interface: {e}")