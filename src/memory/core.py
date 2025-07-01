"""
Core Memory System for HeraAI

Provides the main AdvancedMemorySystem class and public API functions.
This module serves as the primary interface to the memory system.
"""

# CRITICAL: Load NumPy 2.0 compatibility first
try:
    from ..utils.numpy_compat import setup_numpy_compatibility
    setup_numpy_compatibility()
except Exception as e:
    print(f"Warning: NumPy compatibility setup failed in core module: {e}")

import os
import contextlib
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

# Suppress ChromaDB telemetry
class SuppressStderr(contextlib.redirect_stderr):
    def __init__(self):
        super().__init__(open(os.devnull, 'w'))

with SuppressStderr():
    import chromadb
    from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
from collections import defaultdict

from .models import MemoryNode, MemoryCluster, MemoryStatistics, MemoryMetadata, RetrievalResult
from .storage import ChromaDBStorage
from ..config.settings import MEMORY_CONFIG


class AdvancedMemorySystem:
    """
    Advanced Memory System with sophisticated retrieval and anti-hallucination features
    
    This class provides the main interface for memory operations including:
    - User-specific memory storage and retrieval
    - Semantic similarity search with anti-hallucination safeguards
    - Memory clustering and relationship management
    - Comprehensive analytics and validation
    """
    
    def __init__(self):
        """Initialize the advanced memory system"""
        self.config = MEMORY_CONFIG
        self.model: Optional[SentenceTransformer] = None
        self.storage: Optional[ChromaDBStorage] = None
        self.current_user: Optional[str] = None
        self.current_user_id: Optional[str] = None
        
        # In-memory caches
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.clusters: Dict[int, MemoryCluster] = {}
        self.relationship_graph = defaultdict(set)
        self.statistics = MemoryStatistics()
        
        # Initialize components
        self._initialize_model()
        self._initialize_storage()
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model"""
        print("Loading advanced embedding model for enhanced memory...")
        try:
            self.model = SentenceTransformer(self.config["embedding_model"])
            print("✅ Advanced embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading primary model: {e}")
            try:
                self.model = SentenceTransformer(self.config["fallback_model"])
                print("✅ Fallback model loaded")
            except Exception as e2:
                print(f"❌ Error loading fallback model: {e2}")
                raise RuntimeError("Could not load any embedding model")
    
    def _initialize_storage(self) -> None:
        """Initialize the ChromaDB storage with graceful fallback"""
        try:
            self.storage = ChromaDBStorage(self.config["memory_db_path"])
            print("✅ Memory storage initialized")
        except Exception as e:
            print(f"⚠️  Storage initialization failed: {e}")
            print("   Continuing with in-memory storage only")
            print("   💡 To fix: Install hnswlib with: pip install hnswlib")
            self.storage = None
    
    def set_user(self, username: str) -> str:
        """
        Set the current user and load their memory data
        
        Args:
            username: The username to set as current
            
        Returns:
            str: The user ID hash
        """
        self.current_user = username
        self.current_user_id = hashlib.md5(username.encode()).hexdigest()[:16]
        
        # Initialize storage for this user (if available)
        if self.storage:
            collection_name = f"user_memory_{self.current_user_id}"
            self.storage.set_collection(collection_name, {"user": username})
        
        # Load user data
        self._load_user_data()
        self._update_statistics()
        
        storage_status = "with persistent storage" if self.storage else "with in-memory storage only"
        print(f"✅ Memory system ready for user: {username} ({storage_status})")
        return self.current_user_id
    
    def save_memory(self, text: str, role: str, memory_type: str = "conversation",
                   session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save a new memory with advanced processing
        
        Args:
            text: The content to remember
            role: Role of the speaker ('user' or 'ai')
            memory_type: Type of memory (conversation, fact, preference, etc.)
            session_id: Optional session identifier
            context: Optional additional context
            
        Returns:
            Optional[str]: Memory ID if successful, None otherwise
        """
        if not self._ensure_user_set():
            return None
        
        try:
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Analyze importance and emotional context
            importance = self._calculate_importance(text, role, memory_type)
            emotional_context = self._analyze_emotional_context(text)
            
            # Create memory node
            memory_node = MemoryNode.create(
                content=text,
                embedding=embedding,
                role=role,
                memory_type=memory_type,
                user_id=self.current_user_id,
                user_name=self.current_user,
                importance=importance,
                emotional_context=emotional_context,
                session_id=session_id,
                context=context
            )
            
            # Find relationships
            relationships = self._find_relationships(memory_node.id, embedding)
            memory_node.relationships = relationships
            
            # Save to storage (if available)
            storage_saved = False
            if self.storage:
                storage_saved = self.storage.save_memory(memory_node)
            else:
                print("⚠️  Memory saved to in-memory storage only (not persistent)")
                storage_saved = True  # Consider it "saved" for in-memory purposes
            
            if storage_saved:
                # Update local cache
                self.memory_nodes[memory_node.id] = memory_node
                
                # Update statistics
                self.statistics.total_memories += 1
                self.statistics.memory_relationships += len(relationships)
                
                # Trigger clustering if needed
                if len(self.memory_nodes) % self.config["clustering"]["cluster_update_frequency"] == 0:
                    self._update_clusters()
                
                # Save user data
                self._save_user_data()
                
                return memory_node.id
            
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
        
        return None
    
    def retrieve_memories(self, query: str, top_k: int = 10, **filters) -> List[RetrievalResult]:
        """
        Retrieve memories with advanced filtering and anti-hallucination safeguards
        
        Args:
            query: Search query
            top_k: Maximum number of results
            **filters: Additional filters (role, memory_type, etc.)
            
        Returns:
            List[RetrievalResult]: Retrieved memories with metadata
        """
        if not self._ensure_user_set():
            return []
        
        try:
            self.statistics.total_retrievals += 1
            
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in storage (if available)
            raw_results = []
            if self.storage:
                raw_results = self.storage.search_memories(query_embedding, top_k * 2)  # Get more for filtering
            else:
                print("⚠️  Searching in-memory storage only - limited results")
            
            # Process and filter results
            processed_results = []
            for result in raw_results:
                if result['id'] in self.memory_nodes:
                    node = self.memory_nodes[result['id']]
                    
                    # Apply filters
                    if self._passes_filters(node, filters):
                        # Calculate comprehensive scoring
                        retrieval_result = self._create_retrieval_result(
                            node, result, query_embedding
                        )
                        
                        # Apply anti-hallucination filtering
                        if self._passes_quality_threshold(retrieval_result):
                            processed_results.append(retrieval_result)
                            
                            # Update access tracking
                            node.update_access()
            
            # Sort by composite score
            processed_results.sort(key=lambda x: x.composite_score, reverse=True)
            final_results = processed_results[:top_k]
            
            # Update statistics
            if final_results:
                self.statistics.successful_retrievals += 1
            else:
                self.statistics.failed_retrievals += 1
                print(f"🚫 [Anti-hallucination] Filtered out {len(raw_results)} weak matches for '{query[:30]}...'")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            self.statistics.failed_retrievals += 1
            return []
    
    def get_conversation_context(self, query: str, max_length: int = 2500) -> str:
        """
        Get conversation context with strict anti-hallucination filtering
        
        Args:
            query: The query to find relevant context for
            max_length: Maximum length of context string
            
        Returns:
            str: Formatted context string
        """
        memories = self.retrieve_memories(query, top_k=15)
        
        if not memories:
            return ""
        
        context_lines = []
        total_length = 0
        
        for memory in memories:
            if total_length >= max_length:
                break
            
            # Format based on memory type
            metadata = memory.metadata
            mem_type = metadata.get('type', 'conversation')
            role = metadata.get('role', 'user')
            
            if mem_type in ['personal', 'fact', 'preference']:
                line = f"[{mem_type.upper()}] {memory.content}"
            else:
                line = f"[{role.upper()}]: {memory.content}"
            
            if total_length + len(line) <= max_length:
                context_lines.append(line)
                total_length += len(line)
        
        return "\n".join(context_lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return self.statistics.to_dict()
    
    # Private helper methods
    
    def _ensure_user_set(self) -> bool:
        """Ensure a user is currently set"""
        if not self.current_user:
            print("❌ No user set. Please set a user first.")
            return False
        return True
    
    def _calculate_importance(self, text: str, role: str, memory_type: str) -> float:
        """Calculate importance score for a memory"""
        # Use config-based importance calculation
        type_config = self.config["memory_types"].get(memory_type, {})
        base_weight = type_config.get("weight", 1.0)
        
        # Role-based adjustment
        role_weight = 0.7 if role == "user" else 0.3
        
        # Content-based indicators
        importance_indicators = [
            'important', 'remember', 'never forget', 'always', 'crucial', 'essential',
            'my name is', 'i am', 'i love', 'i hate', 'my goal', 'my dream'
        ]
        
        content_score = sum(1 for indicator in importance_indicators if indicator in text.lower())
        content_weight = min(content_score * 0.1, 0.5)
        
        # Length-based importance
        length_weight = min(len(text) / 1000, 0.2)
        
        total_score = (base_weight + role_weight + content_weight + length_weight) / 4
        return min(total_score, 1.0)
    
    def _analyze_emotional_context(self, text: str) -> Dict[str, float]:
        """Analyze emotional context of text"""
        emotion_keywords = {
            'joy': ['happy', 'excited', 'glad', 'pleased', 'delighted', 'wonderful'],
            'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'unhappy'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected']
        }
        
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions[emotion] = min(score / len(keywords), 1.0)
        
        return emotions
    
    def _find_relationships(self, memory_id: str, embedding: List[float]) -> List[str]:
        """Find relationships with existing memories"""
        relationships = []
        threshold = self.config["semantic_similarity_threshold"]
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        for existing_id, node in self.memory_nodes.items():
            if existing_id != memory_id:
                similarity = cosine_similarity([embedding], [node.embedding])[0][0]
                if similarity > threshold:
                    relationships.append(existing_id)
                    # Add bidirectional relationship
                    self.relationship_graph[memory_id].add(existing_id)
                    self.relationship_graph[existing_id].add(memory_id)
        
        return relationships
    
    def _passes_filters(self, node: MemoryNode, filters: Dict[str, Any]) -> bool:
        """Check if memory node passes the given filters"""
        for key, value in filters.items():
            if key == 'role' and node.metadata.role != value:
                return False
            elif key == 'memory_type' and node.metadata.memory_type != value:
                return False
            elif key == 'min_importance' and node.importance_score < value:
                return False
        return True
    
    def _create_retrieval_result(self, node: MemoryNode, storage_result: Dict[str, Any], 
                               query_embedding: List[float]) -> RetrievalResult:
        """Create a retrieval result with comprehensive scoring"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate semantic similarity
        semantic_similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
        
        # Calculate time decay
        memory_time = datetime.fromisoformat(node.metadata.timestamp)
        time_diff = datetime.utcnow() - memory_time
        time_decay = max(0.1, 1.0 - (time_diff.days / 365.0))
        
        # Calculate composite score
        composite_score = (
            semantic_similarity * 0.4 +
            node.importance_score * 0.3 +
            time_decay * 0.2 +
            min(node.access_count * 0.05, 0.1)
        )
        
        return RetrievalResult.from_memory_node(
            node=node,
            relevance_score=storage_result.get('distance', 0),
            semantic_similarity=semantic_similarity,
            composite_score=composite_score,
            time_decay=time_decay,
            retrieval_strategy="semantic"
        )
    
    def _passes_quality_threshold(self, result: RetrievalResult) -> bool:
        """Apply anti-hallucination quality thresholds"""
        config = self.config
        
        # Strong semantic similarity (main filter)
        if result.semantic_similarity > config["strict_semantic_threshold"]:
            return True
        
        # Important personal facts with some relevance
        if (result.semantic_similarity > config["min_access_semantic_threshold"] and 
            result.importance_score > config["min_importance_for_weak_matches"] and
            result.metadata.get('type') in ['personal', 'fact']):
            return True
        
        # Previously discussed topics (high access count + some relevance)
        if (result.semantic_similarity > config["min_access_semantic_threshold"] and 
            result.access_count > config["access_count_threshold"]):
            return True
        
        return False
    
    def _load_user_data(self) -> None:
        """Load user-specific memory data from storage"""
        # This would load cached data from files
        # For now, we'll rebuild from storage
        pass
    
    def _save_user_data(self) -> None:
        """Save user-specific memory data to storage"""
        # This would save cached data to files
        # For now, everything is saved to ChromaDB
        pass
    
    def _update_clusters(self) -> None:
        """Update memory clusters"""
        # Implementation would go here
        pass
    
    def _update_statistics(self) -> None:
        """Update memory statistics"""
        if self.storage and self.storage.collection:
            try:
                count = self.storage.collection.count()
                self.statistics.total_memories = count
            except:
                pass


# Global instance for backward compatibility
_memory_system = AdvancedMemorySystem()

# Public API functions
def set_user(username: str) -> str:
    """Set the current user for memory operations"""
    return _memory_system.set_user(username)

def save_memory(text: str, role: str, mem_type: str = "conversation") -> Optional[str]:
    """Save a memory for the current user"""
    return _memory_system.save_memory(text, role, mem_type)

def retrieve_memories(query: str, top_k: int = 10, role: str = None, mem_type: str = None) -> List[str]:
    """Retrieve memories for the current user (backward compatible)"""
    filters = {}
    if role:
        filters['role'] = role
    if mem_type:
        filters['memory_type'] = mem_type
    
    results = _memory_system.retrieve_memories(query, top_k, **filters)
    return [result.content for result in results]

def get_conversation_context(query: str, max_length: int = 2500) -> str:
    """Get conversation context for the current user"""
    return _memory_system.get_conversation_context(query, max_length)

def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics for the current user"""
    return _memory_system.get_statistics()

def retrieve_memories_advanced(query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Advanced retrieve function with full metadata"""
    results = _memory_system.retrieve_memories(query, top_k, **kwargs)
    return [result.to_dict() for result in results] 