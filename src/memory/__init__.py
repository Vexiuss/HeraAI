"""
Advanced Memory System for HeraAI

This package provides sophisticated memory management capabilities including:
- Long-term memory storage and retrieval
- Semantic similarity clustering  
- Memory relationships and analytics
- Anti-hallucination safeguards
- User-specific memory isolation
"""

from .core import AdvancedMemorySystem
from .models import MemoryNode, MemoryCluster, MemoryMetadata
from .storage import ChromaDBStorage
# Note: clustering, relationships, and analytics modules are planned for future implementation

# Convenience imports for the public API
from .core import (
    set_user,
    save_memory, 
    retrieve_memories,
    get_conversation_context,
    get_memory_stats,
    retrieve_memories_advanced
)

__all__ = [
    # Core classes
    'AdvancedMemorySystem',
    'MemoryNode', 
    'MemoryCluster',
    'MemoryMetadata',
    'ChromaDBStorage',
    
    # Public API functions
    'set_user',
    'save_memory',
    'retrieve_memories', 
    'get_conversation_context',
    'get_memory_stats',
    'retrieve_memories_advanced'
] 