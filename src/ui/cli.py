"""
Command Line Interface for HeraAI

Provides a clean CLI for interacting with the voice assistant.
"""

from typing import List, Dict, Any
import sys
from datetime import datetime


class CLIInterface:
    """Command-line interface utilities for HeraAI"""
    
    @staticmethod
    def print_banner():
        """Print the HeraAI welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🤖 HERA AI - Advanced Voice Assistant v2.0                ║
║                                                               ║
║    🧠 Advanced Memory System                                  ║
║    🎤 Natural Voice Interaction                               ║
║    🔮 Personalized Conversations                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    @staticmethod
    def print_help():
        """Print available commands and help"""
        help_text = """
🆘 === HERAAI HELP ===

📢 Voice Commands:
   • Speak naturally - I'll detect when you finish
   • Say something during my response to interrupt
   
⌨️  Text Commands:
   • 'help' or '?' - Show this help
   • 'stats' - Show memory statistics
   • 'profile' - Show user profile
   • 'users' - List all users
   • 'switch user' - Change to different user
   • 'memory test' - Test memory functionality
   • 'voice test' - Test microphone
   • 'clear' - Clear conversation history
   • 'debug on/off' - Toggle debug mode
   • 'quit' or 'exit' - Exit the application

🎛️  Advanced Commands:
   • 'memory insights' - AI memory analysis
   • 'memory search <query>' - Advanced memory search
   • 'memory report' - Generate memory report
   • 'memory health' - Check memory system health
   • 'visualize' - Create memory visualizations
   
📝 Example Conversations:
   • "What did I tell you about my hobbies?"
   • "Remember that I love pizza"
   • "What are my preferences?"
        """
        print(help_text)
    
    @staticmethod
    def print_memory_stats(stats: Dict[str, Any]):
        """
        Print formatted memory statistics
        
        Args:
            stats: Memory statistics dictionary
        """
        print(f"\n📊 === Memory Statistics ===")
        print(f"🧠 Total memories: {stats.get('total_memories', 0)}")
        print(f"🔍 Total retrievals: {stats.get('total_retrievals', 0)}")
        print(f"✅ Successful retrievals: {stats.get('successful_retrievals', 0)}")
        print(f"❌ Failed retrievals: {stats.get('failed_retrievals', 0)}")
        print(f"🗂️ Memory clusters: {stats.get('memory_clusters', 0)}")
        print(f"📊 Avg cluster size: {stats.get('avg_cluster_size', 0):.1f}")
        print(f"🔗 Memory relationships: {stats.get('memory_relationships', 0)}")
        
        # Calculate success rate
        total_retrievals = stats.get('total_retrievals', 0)
        if total_retrievals > 0:
            success_rate = (stats.get('successful_retrievals', 0) / total_retrievals) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
    
    @staticmethod
    def print_conversation_start(username: str):
        """
        Print conversation start message
        
        Args:
            username: The current user's name
        """
        print(f"\n💬 === Starting Conversation with {username} ===")
        print("🎤 I'm listening... Speak naturally, and I'll respond!")
        print("💡 Type 'help' for commands or just start talking!")
        print("🛑 Say something while I'm speaking to interrupt me")
        print("-" * 60)
    
    @staticmethod
    def print_memory_context_info(context_available: bool, context_length: int = 0):
        """
        Print information about available memory context
        
        Args:
            context_available: Whether memory context is available
            context_length: Length of the context
        """
        if context_available:
            print(f"📝 [Retrieved {context_length} relevant memories]")
        else:
            print("📝 [No relevant memories found - treating as new conversation]")
    
    @staticmethod
    def print_debug_info(message: str):
        """
        Print debug information with timestamp
        
        Args:
            message: Debug message to print
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"🐛 [{timestamp}] {message}")
    
    @staticmethod
    def print_error(message: str):
        """
        Print error message in a formatted way
        
        Args:
            message: Error message to print
        """
        print(f"❌ Error: {message}")
    
    @staticmethod
    def print_warning(message: str):
        """
        Print warning message in a formatted way
        
        Args:
            message: Warning message to print
        """
        print(f"⚠️ Warning: {message}")
    
    @staticmethod
    def print_success(message: str):
        """
        Print success message in a formatted way
        
        Args:
            message: Success message to print
        """
        print(f"✅ {message}")
    
    @staticmethod
    def print_info(message: str):
        """
        Print info message in a formatted way
        
        Args:
            message: Info message to print
        """
        print(f"ℹ️ {message}")
    
    @staticmethod
    def confirm_action(message: str) -> bool:
        """
        Ask user for confirmation
        
        Args:
            message: Confirmation message
            
        Returns:
            bool: True if user confirms, False otherwise
        """
        response = input(f"❓ {message} (y/N): ").strip().lower()
        return response in ['y', 'yes']
    
    @staticmethod
    def get_user_input(prompt: str) -> str:
        """
        Get user input with a formatted prompt
        
        Args:
            prompt: The prompt message
            
        Returns:
            str: User's input
        """
        return input(f"💭 {prompt}: ").strip()
    
    @staticmethod
    def print_system_info():
        """Print system information"""
        print("\n🔧 === System Information ===")
        print(f"🐍 Python version: {sys.version.split()[0]}")
        print(f"💻 Platform: {sys.platform}")
        print("📦 Required packages: speech_recognition, edge-tts, pygame, requests")
        print("🤖 AI Backend: Ollama (requires local installation)")
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_conversation_history(conversation: List[Dict[str, str]], max_lines: int = 10):
        """
        Print recent conversation history
        
        Args:
            conversation: List of conversation messages
            max_lines: Maximum number of lines to show
        """
        print(f"\n💬 === Recent Conversation (last {max_lines} messages) ===")
        
        recent_messages = conversation[-max_lines:] if len(conversation) > max_lines else conversation
        
        for message in recent_messages:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            if role == 'user':
                print(f"👤 You: {content}")
            elif role == 'ai':
                print(f"🤖 AI: {content}")
            else:
                print(f"❓ {role}: {content}")
    
    @staticmethod
    def print_advanced_memory_info(memories: List[Dict[str, Any]]):
        """
        Print detailed information about retrieved memories
        
        Args:
            memories: List of memory objects with metadata
        """
        if not memories:
            print("📝 No memories found")
            return
        
        print(f"\n🧠 === Advanced Memory Details ({len(memories)} memories) ===")
        
        for i, memory in enumerate(memories[:5], 1):  # Show first 5
            content = memory.get('text', memory.get('content', ''))[:80] + "..."
            relevance = memory.get('semantic_similarity', 0)
            importance = memory.get('importance_score', 0)
            mem_type = memory.get('metadata', {}).get('type', 'unknown')
            
            print(f"{i}. [{mem_type.upper()}] {content}")
            print(f"   Relevance: {relevance:.3f} | Importance: {importance:.3f}")
        
        if len(memories) > 5:
            print(f"   ... and {len(memories) - 5} more memories") 