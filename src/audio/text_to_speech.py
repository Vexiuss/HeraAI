"""
Text-to-Speech Module for HeraAI

Provides realistic text-to-speech using Edge TTS with voice interruption capabilities.
"""

import asyncio
import edge_tts
import pygame
import time
import threading
import os
from typing import Callable, Optional
from ..config.settings import AUDIO_CONFIG


class TextToSpeechEngine:
    """Enhanced TTS with voice-based interruption"""
    
    def __init__(self, interruption_callback: Optional[Callable[[], bool]] = None):
        """
        Initialize the TTS engine
        
        Args:
            interruption_callback: Function that returns True if TTS should be interrupted
        """
        self.config = AUDIO_CONFIG["tts"]
        self.interruption_callback = interruption_callback
        self.interrupt_tts = False
        
    def speak(self, text: str) -> None:
        """
        Convert text to speech with interruption support
        
        Args:
            text: The text to speak
        """
        self.interrupt_tts = False
        print(f"🤖 AI: {text}")
        
        # Generate speech file
        if not self._generate_speech_file(text):
            return
        
        # Play audio with interruption monitoring
        self._play_with_interruption()
    
    def _generate_speech_file(self, text: str) -> bool:
        """
        Generate speech file using Edge TTS
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bool: True if file generated successfully, False otherwise
        """
        try:
            async def _generate():
                communicate = edge_tts.Communicate(text, self.config["voice"])
                await communicate.save(self.config["output_file"])
            
            asyncio.run(_generate())
            return True
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return False
    
    def _play_with_interruption(self) -> None:
        """Play audio with interruption monitoring"""
        # Shared events for thread communication
        audio_playing = threading.Event()
        audio_finished = threading.Event()
        
        def play_audio():
            """Audio playback thread"""
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(self.config["output_file"])
                pygame.mixer.music.play()
                audio_playing.set()
                
                # Monitor playback
                while pygame.mixer.music.get_busy() and not self.interrupt_tts:
                    time.sleep(self.config["audio_check_interval"])
                
                if self.interrupt_tts:
                    pygame.mixer.music.stop()
                    print("🛑 Audio interrupted by voice")
                
                pygame.mixer.quit()
                
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
                
            finally:
                audio_finished.set()
                self._cleanup_audio_file()
        
        def monitor_interruption():
            """Monitor for voice interruption"""
            # Wait for audio to start
            audio_playing.wait(timeout=self.config["audio_start_timeout"])
            
            while not audio_finished.is_set() and not self.interrupt_tts:
                if self.interruption_callback and self.interruption_callback():
                    self.interrupt_tts = True
                    break
                time.sleep(self.config["interrupt_check_interval"])
        
        # Start audio playback
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()
        
        # Start interruption monitoring if callback provided
        if self.interruption_callback:
            print("🎤 Say something to interrupt...")
            interrupt_monitor = threading.Thread(target=monitor_interruption, daemon=True)
            interrupt_monitor.start()
        
        # Wait for audio to finish
        audio_thread.join()
    
    def _cleanup_audio_file(self) -> None:
        """Clean up the temporary audio file"""
        try:
            if os.path.exists(self.config["output_file"]):
                os.remove(self.config["output_file"])
        except Exception as e:
            print(f"⚠️ Could not clean up audio file: {e}")
    
    def stop(self) -> None:
        """Force stop current TTS playback"""
        self.interrupt_tts = True
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        self._cleanup_audio_file()
    
    def set_voice(self, voice: str) -> None:
        """
        Change the TTS voice
        
        Args:
            voice: Edge TTS voice identifier (e.g., "en-US-JennyNeural")
        """
        self.config["voice"] = voice
        print(f"🔧 TTS voice changed to: {voice}")
    
    @staticmethod
    def list_available_voices() -> list:
        """
        List available Edge TTS voices
        
        Returns:
            list: Available voice identifiers
        """
        # Common Edge TTS voices
        voices = [
            "en-US-JennyNeural",
            "en-US-GuyNeural", 
            "en-US-AriaNeural",
            "en-US-DavisNeural",
            "en-US-AmberNeural",
            "en-US-AnaNeural",
            "en-US-AshleyNeural",
            "en-US-BrandonNeural",
            "en-US-ChristopherNeural",
            "en-US-CoraNeural"
        ]
        return voices 