"""
Audio System for HeraAI

This package provides audio input/output capabilities including:
- Speech recognition with natural pause detection
- Text-to-speech with voice interruption
- Voice-based interaction management
"""

from .speech_recognition import VoiceRecognizer
from .text_to_speech import TextToSpeechEngine
from .voice_interruption import VoiceInterruptionManager

__all__ = ['VoiceRecognizer', 'TextToSpeechEngine', 'VoiceInterruptionManager'] 