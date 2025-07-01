"""
Voice Interruption Module for HeraAI

Provides voice-based interruption detection during TTS playback.
"""

import speech_recognition as sr
from typing import Optional
from ..config.settings import AUDIO_CONFIG


class VoiceInterruptionManager:
    """Manages voice-based interruption of TTS"""
    
    def __init__(self):
        """Initialize the voice interruption manager"""
        self.recognizer = sr.Recognizer()
        self.config = AUDIO_CONFIG["interruption"]
    
    def detect_voice_interrupt(self) -> bool:
        """
        Detect if user starts speaking to interrupt TTS
        
        Returns:
            bool: True if voice detected, False otherwise
        """
        try:
            with sr.Microphone() as source:
                # Quick ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(
                    source, 
                    duration=self.config["ambient_noise_duration"]
                )
                
                # Configure for interrupt detection (more sensitive)
                self.recognizer.pause_threshold = self.config["pause_threshold"]
                self.recognizer.energy_threshold = self.config["energy_threshold"]
                self.recognizer.non_speaking_duration = self.config["non_speaking_duration"]
                
                try:
                    # Short listen period to detect voice start
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.config["detection_timeout"], 
                        phrase_time_limit=self.config["phrase_time_limit"]
                    )
                    # If we get here, voice was detected
                    print("🛑 Voice detected - interrupting AI...")
                    return True
                    
                except sr.WaitTimeoutError:
                    # No voice detected, continue
                    return False
                    
        except Exception:
            # On any error, don't interrupt
            return False
    
    def wait_for_voice_interrupt(self, check_interval: float = 0.1) -> bool:
        """
        Continuously check for voice interruption
        
        Args:
            check_interval: How often to check for voice (seconds)
            
        Returns:
            bool: True if voice was detected, False if stopped for other reasons
        """
        import time
        
        try:
            while True:
                if self.detect_voice_interrupt():
                    return True
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            return False
    
    def adjust_sensitivity(self, energy_threshold: int = None) -> None:
        """
        Adjust interruption detection sensitivity
        
        Args:
            energy_threshold: Minimum energy level to consider as voice
        """
        if energy_threshold is not None:
            self.config["energy_threshold"] = energy_threshold
            print(f"🔧 Interruption sensitivity adjusted to {energy_threshold}")
    
    def test_interruption_detection(self) -> None:
        """Test the voice interruption detection"""
        print("🎤 Testing voice interruption detection...")
        print("Speak now to test interruption detection...")
        
        if self.detect_voice_interrupt():
            print("✅ Voice interruption detection working")
        else:
            print("⚠️ No voice detected - try speaking louder or adjusting sensitivity") 