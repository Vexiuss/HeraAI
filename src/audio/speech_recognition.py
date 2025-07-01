"""
Speech Recognition Module for HeraAI

Provides enhanced speech recognition with natural pause detection
and ambient noise adjustment.
"""

import speech_recognition as sr
from typing import Optional
from ..config.settings import AUDIO_CONFIG


class VoiceRecognizer:
    """Enhanced speech recognition with natural pause detection"""
    
    def __init__(self):
        """Initialize the voice recognizer with configuration"""
        self.recognizer = sr.Recognizer()
        self.config = AUDIO_CONFIG["speech_recognition"]
        self._configure_recognizer()
    
    def _configure_recognizer(self) -> None:
        """Configure the speech recognizer with optimal settings"""
        self.recognizer.pause_threshold = self.config["pause_threshold"]
        self.recognizer.energy_threshold = self.config["energy_threshold"]
        self.recognizer.dynamic_energy_threshold = self.config["dynamic_energy_threshold"]
        self.recognizer.dynamic_energy_adjustment_damping = self.config["dynamic_energy_adjustment_damping"]
        self.recognizer.dynamic_energy_ratio = self.config["dynamic_energy_ratio"]
        self.recognizer.phrase_time_limit = self.config["phrase_time_limit"]
        self.recognizer.non_speaking_duration = self.config["non_speaking_duration"]
    
    def listen(self) -> Optional[str]:
        """
        Listen for speech input with natural pause detection
        
        Returns:
            Optional[str]: Recognized text or None if no speech detected
        """
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                print("🎤 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(
                    source, 
                    duration=self.config["ambient_noise_duration"]
                )
                
                print("🗣️ Listening... (speak naturally, I'll detect when you finish)")
                
                try:
                    # Listen with timeout but allow for natural pauses
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.config["timeout"], 
                        phrase_time_limit=self.config["phrase_time_limit"]
                    )
                except sr.WaitTimeoutError:
                    print("⏰ No speech detected. Please try again.")
                    return None
            
            # Process the audio
            return self._process_audio(audio)
            
        except Exception as e:
            print(f"❌ Error in speech recognition setup: {e}")
            return None
    
    def _process_audio(self, audio: sr.AudioData) -> Optional[str]:
        """
        Process audio data and convert to text
        
        Args:
            audio: The recorded audio data
            
        Returns:
            Optional[str]: Recognized text or None if recognition failed
        """
        try:
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"✅ You said: {text}")
            return text
            
        except sr.UnknownValueError:
            print("❌ Sorry, I couldn't understand that. Please speak more clearly.")
            return None
            
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def adjust_sensitivity(self, energy_threshold: int = None, 
                          pause_threshold: float = None) -> None:
        """
        Adjust recognition sensitivity on the fly
        
        Args:
            energy_threshold: Minimum audio energy threshold
            pause_threshold: Pause duration before considering speech finished
        """
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
            print(f"🔧 Adjusted energy threshold to {energy_threshold}")
        
        if pause_threshold is not None:
            self.recognizer.pause_threshold = pause_threshold
            print(f"🔧 Adjusted pause threshold to {pause_threshold}")
    
    def test_microphone(self) -> bool:
        """
        Test if microphone is working properly
        
        Returns:
            bool: True if microphone is working, False otherwise
        """
        try:
            with sr.Microphone() as source:
                print("🎤 Testing microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone test successful")
                return True
                
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False 