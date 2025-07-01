#!/usr/bin/env python3
"""
HeraAI - Advanced Voice Assistant v2.0

A sophisticated voice assistant with advanced memory capabilities,
natural voice interaction, and personalized conversations.

This is the refactored version using a clean modular architecture.
"""

import os
import sys
import warnings

# Completely disable ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_SERVER_AUTH_CREDENTIALS_FILE'] = ''
os.environ['CHROMA_SERVER_AUTH_CREDENTIALS'] = ''
os.environ['CHROMA_SERVER_AUTH_PROVIDER'] = ''

# Suppress ChromaDB telemetry warnings globally
warnings.filterwarnings("ignore", message=".*capture.*takes.*positional argument.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

# CRITICAL: Import NumPy compatibility layer before any other imports
# This fixes ChromaDB compatibility with NumPy 2.0+
try:
    from src.utils.numpy_compat import setup_numpy_compatibility, patch_chromadb_types
    patch_chromadb_types()  # Pre-patch ChromaDB before it's imported
    print("✅ NumPy 2.0 compatibility layer activated")
except ImportError as e:
    print(f"⚠️  Warning: Could not load NumPy compatibility layer: {e}")
except Exception as e:
    print(f"⚠️  Warning: NumPy compatibility setup failed: {e}")
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import the refactored HeraAI modules
try:
    from src.config.settings import AI_CONFIG, AUDIO_CONFIG, SYSTEM_CONFIG
    from src.memory.core import AdvancedMemorySystem, set_user, save_memory, get_conversation_context
    from src.audio.speech_recognition import VoiceRecognizer
    from src.audio.text_to_speech import TextToSpeechEngine
    from src.audio.voice_interruption import VoiceInterruptionManager
    from src.ai.ollama_client import OllamaClient
    from src.ui.user_management import UserManager
    from src.ui.cli import CLIInterface
    from src.utils.logging import setup_logger
    from src.utils.error_handling import handle_exceptions, HeraAIException
except ImportError as e:
    print(f"❌ Error importing HeraAI modules: {e}")
    print("Please ensure all required dependencies are installed and the src directory exists.")
    sys.exit(1)


class HeraAI:
    """
    Main HeraAI application class
    
    Orchestrates all the components to provide a seamless voice assistant experience.
    """
    
    def __init__(self):
        """Initialize the HeraAI application"""
        self.logger = setup_logger("heraai.main")
        self.cli = CLIInterface()
        
        # Initialize components
        self.memory_system: Optional[AdvancedMemorySystem] = None
        self.voice_recognizer: Optional[VoiceRecognizer] = None
        self.tts_engine: Optional[TextToSpeechEngine] = None
        self.voice_interruptor: Optional[VoiceInterruptionManager] = None
        self.ai_client: Optional[OllamaClient] = None
        self.user_manager: Optional[UserManager] = None
        
        # Application state
        self.current_user: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.debug_mode: bool = SYSTEM_CONFIG.get("debug", False)
        self.running: bool = False
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all HeraAI components"""
        try:
            self.cli.print_banner()
            print("🚀 Initializing HeraAI components...")
            
            # Initialize memory system
            print("🧠 Initializing memory system...")
            self.memory_system = AdvancedMemorySystem()
            
            # Initialize voice components
            print("🎤 Initializing voice recognition...")
            self.voice_recognizer = VoiceRecognizer()
            
            print("🔊 Initializing text-to-speech...")
            self.voice_interruptor = VoiceInterruptionManager()
            self.tts_engine = TextToSpeechEngine(
                interruption_callback=self.voice_interruptor.detect_voice_interrupt
            )
            
            # Initialize AI client
            print("🤖 Initializing AI client...")
            self.ai_client = OllamaClient()
            
            # Test AI connection
            if not self.ai_client.check_connection():
                self.cli.print_warning("Ollama AI service not detected. Please ensure it's running.")
                self.cli.print_info("Install Ollama from: https://ollama.ai")
            
            # Initialize user management
            print("👤 Initializing user management...")
            self.user_manager = UserManager()
            
            print("✅ All components initialized successfully!")
            
        except Exception as e:
            self.cli.print_error(f"Failed to initialize components: {e}")
            self.logger.error(f"Initialization error: {e}")
            raise HeraAIException(f"Component initialization failed: {e}")
    
    def start(self) -> None:
        """Start the HeraAI application"""
        try:
            self.running = True
            
            # User identification
            self.current_user = self.user_manager.identify_user()
            set_user(self.current_user)
            
            # Start conversation
            self.cli.print_conversation_start(self.current_user)
            
            # Main conversation loop
            self._conversation_loop()
            
        except KeyboardInterrupt:
            self.cli.print_info("Goodbye! 👋")
        except Exception as e:
            self.cli.print_error(f"Application error: {e}")
            self.logger.error(f"Application error: {e}")
        finally:
            self.running = False
    
    def _conversation_loop(self) -> None:
        """Main conversation loop"""
        while self.running:
            try:
                # Get user input (voice or text)
                user_input = self._get_user_input()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if self._handle_command(user_input):
                    continue
                
                # Process conversation
                self._process_conversation(user_input)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.cli.print_error(f"Error in conversation loop: {e}")
                if self.debug_mode:
                    self.logger.error(f"Conversation error: {e}")
    
    def _get_user_input(self) -> Optional[str]:
        """Get input from user (voice or text)"""
        print("\n" + "="*60)
        print("🎤 Listening for your voice... (or type 'text' for text input)")
        
        # Try voice input first
        user_input = self.voice_recognizer.listen()
        
        # Fallback to text input
        if not user_input:
            text_input = input("💬 Type your message (or press Enter to try voice again): ").strip()
            if text_input:
                user_input = text_input
        
        return user_input
    
    def _handle_command(self, user_input: str) -> bool:
        """
        Handle special commands
        
        Args:
            user_input: The user's input
            
        Returns:
            bool: True if command was handled, False for normal conversation
        """
        command = user_input.lower().strip()
        
        # Help commands
        if command in ['help', '?', 'commands']:
            self.cli.print_help()
            return True
        
        # Exit commands
        if command in ['quit', 'exit', 'bye', 'goodbye']:
            self.cli.print_info("Goodbye! 👋")
            self.running = False
            return True
        
        # Memory statistics
        if command in ['stats', 'statistics', 'memory stats']:
            stats = self.memory_system.get_statistics()
            self.cli.print_memory_stats(stats)
            return True
        
        # User profile
        if command in ['profile', 'user profile', 'my profile']:
            self.user_manager.display_user_stats()
            return True
        
        # List users
        if command in ['users', 'list users', 'show users']:
            users = self.user_manager.list_users()
            print("\n👥 === All Users ===")
            for user in users:
                print(f"   • {user['name']} (Last seen: {user.get('last_seen', 'Never')})")
            return True
        
        # Switch user
        if command in ['switch user', 'change user', 'switch']:
            new_user = self.user_manager.identify_user()
            self.current_user = new_user
            set_user(new_user)
            self.conversation_history.clear()
            self.cli.print_success(f"Switched to user: {new_user}")
            return True
        
        # Clear conversation
        if command in ['clear', 'clear conversation', 'reset']:
            self.conversation_history.clear()
            self.cli.print_success("Conversation history cleared")
            return True
        
        # Debug mode toggle
        if command in ['debug on', 'debug true']:
            self.debug_mode = True
            self.cli.print_success("Debug mode enabled")
            return True
        
        if command in ['debug off', 'debug false']:
            self.debug_mode = False
            self.cli.print_success("Debug mode disabled")
            return True
        
        # Voice test
        if command in ['voice test', 'test microphone', 'mic test']:
            if self.voice_recognizer.test_microphone():
                self.cli.print_success("Microphone is working properly")
            else:
                self.cli.print_error("Microphone test failed")
            return True
        
        # Memory test
        if command in ['memory test', 'test memory']:
            self._test_memory_system()
            return True
        
        # System info
        if command in ['system info', 'info', 'version']:
            self.cli.print_system_info()
            return True
        
        # Show recent conversation
        if command in ['history', 'conversation history', 'recent']:
            self.cli.print_conversation_history(self.conversation_history)
            return True
        
        return False
    
    def _process_conversation(self, user_input: str) -> None:
        """
        Process a normal conversation exchange
        
        Args:
            user_input: The user's input to process
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save user input to memory
            save_memory(user_input, "user", "conversation")
            
            # Get relevant context from memory
            context = get_conversation_context(user_input)
            has_context = bool(context.strip())
            
            if self.debug_mode:
                self.cli.print_memory_context_info(has_context, len(context))
            
            # Build recent conversation context
            recent_messages = self.conversation_history[-AI_CONFIG["conversation"]["recent_messages_count"]:]
            recent_context = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            # Build AI prompt
            prompt = self.ai_client.build_conversation_prompt(
                user_input=user_input,
                context=context,
                recent_conversation=recent_context,
                has_memory_context=has_context
            )
            
            if self.debug_mode:
                self.cli.print_debug_info(f"Prompt length: {len(prompt)} characters")
            
            # Generate AI response
            ai_response = self.ai_client.generate_response(prompt)
            
            # Add AI response to conversation history
            self.conversation_history.append({
                'role': 'ai',
                'content': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save AI response to memory
            save_memory(ai_response, "ai", "conversation")
            
            # Speak the response
            self.tts_engine.speak(ai_response)
            
        except Exception as e:
            self.cli.print_error(f"Error processing conversation: {e}")
            if self.debug_mode:
                self.logger.error(f"Conversation processing error: {e}")
    
    def _test_memory_system(self) -> None:
        """Test the memory system functionality"""
        print("\n🧪 === Testing Memory System ===")
        
        try:
            # Test saving a memory
            test_memory = "This is a test memory for system validation"
            memory_id = save_memory(test_memory, "user", "test")
            
            if memory_id:
                self.cli.print_success("Memory save test passed")
            else:
                self.cli.print_error("Memory save test failed")
                return
            
            # Test retrieving memories
            retrieved = self.memory_system.retrieve_memories("test memory")
            
            if retrieved:
                self.cli.print_success(f"Memory retrieval test passed ({len(retrieved)} results)")
            else:
                self.cli.print_warning("Memory retrieval test returned no results")
            
            # Test context generation
            context = get_conversation_context("test")
            
            if context:
                self.cli.print_success("Context generation test passed")
            else:
                self.cli.print_info("Context generation test returned empty context")
            
            # Show statistics
            stats = self.memory_system.get_statistics()
            self.cli.print_memory_stats(stats)
            
        except Exception as e:
            self.cli.print_error(f"Memory system test failed: {e}")
    
    def stop(self) -> None:
        """Stop the HeraAI application"""
        self.running = False
        if self.tts_engine:
            self.tts_engine.stop()
        
        self.cli.print_info("HeraAI application stopped")


# Advanced features (if available)
def check_advanced_features():
    """Check if advanced memory features are available"""
    try:
        from memory_enhancements import (
            get_ai_memory_insights, create_memory_visualizations,
            predict_future_memories, search_memories_advanced,
            generate_memory_report
        )
        return True
    except ImportError:
        return False


def main():
    """Main entry point for HeraAI"""
    try:
        # Check for advanced features
        if check_advanced_features():
            print("✅ Advanced memory features detected")
        else:
            print("ℹ️ Running with standard memory features")
        
        # Create and start HeraAI
        hera = HeraAI()
        hera.start()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 