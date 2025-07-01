"""
HeraAI - Advanced Voice Assistant with Long-Term Memory

A sophisticated voice assistant that uses advanced memory systems to provide
personalized and contextually aware conversations.
"""

__version__ = "2.0.0"
__author__ = "HeraAI Development Team"

# CRITICAL: Load NumPy 2.0 compatibility layer first
try:
    from .utils.numpy_compat import setup_numpy_compatibility
    setup_numpy_compatibility()
    print("✅ NumPy 2.0 compatibility layer loaded for HeraAI package")
except Exception as e:
    print(f"⚠️  Warning: NumPy compatibility setup failed: {e}")

from .config.settings import MEMORY_CONFIG, AUDIO_CONFIG, AI_CONFIG
from .memory.core import AdvancedMemorySystem
from .audio.speech_recognition import VoiceRecognizer
from .audio.text_to_speech import TextToSpeechEngine
from .ai.ollama_client import OllamaClient
from .ui.user_management import UserManager

__all__ = [
    'MEMORY_CONFIG',
    'AUDIO_CONFIG', 
    'AI_CONFIG',
    'AdvancedMemorySystem',
    'VoiceRecognizer',
    'TextToSpeechEngine',
    'OllamaClient',
    'UserManager'
] 