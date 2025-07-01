"""
Data models for the HeraAI memory system

This module defines the core data structures used throughout the memory system,
providing clear interfaces and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import uuid
import json


@dataclass
class MemoryMetadata:
    """Metadata for a memory entry"""
    role: str
    memory_type: str
    timestamp: str
    session_id: str
    importance: float
    user_id: str
    user_name: str
    emotional_context: str = "{}"
    context: str = "{}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "role": self.role,
            "type": self.memory_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "importance": self.importance,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "emotional_context": self.emotional_context,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        """Create from dictionary"""
        return cls(
            role=data.get("role", "user"),
            memory_type=data.get("type", "conversation"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            session_id=data.get("session_id", ""),
            importance=data.get("importance", 0.5),
            user_id=data.get("user_id", ""),
            user_name=data.get("user_name", ""),
            emotional_context=data.get("emotional_context", "{}"),
            context=data.get("context", "{}")
        )
    
    def get_emotional_context(self) -> Dict[str, float]:
        """Parse emotional context from JSON string"""
        try:
            return json.loads(self.emotional_context)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_emotional_context(self, emotions: Dict[str, float]) -> None:
        """Set emotional context as JSON string"""
        self.emotional_context = json.dumps(emotions)


@dataclass
class MemoryNode:
    """A single memory node containing content and metadata"""
    id: str
    content: str
    embedding: List[float]
    metadata: MemoryMetadata
    relationships: List[str] = field(default_factory=list)
    cluster_id: Optional[int] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: str = ""
    emotional_context: Dict[str, float] = field(default_factory=dict)
    validation_score: float = 1.0
    
    @classmethod
    def create(cls, content: str, embedding: List[float], 
               role: str, memory_type: str, user_id: str, user_name: str,
               importance: float, emotional_context: Dict[str, float] = None,
               session_id: str = None, context: Dict[str, Any] = None) -> 'MemoryNode':
        """Factory method to create a new memory node"""
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        metadata = MemoryMetadata(
            role=role,
            memory_type=memory_type,
            timestamp=timestamp,
            session_id=session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            importance=importance,
            user_id=user_id,
            user_name=user_name,
            emotional_context=json.dumps(emotional_context or {}),
            context=json.dumps(context or {})
        )
        
        return cls(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            importance_score=importance,
            last_accessed=timestamp,
            emotional_context=emotional_context or {}
        )
    
    def update_access(self) -> None:
        """Update access tracking information"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow().isoformat()
    
    def add_relationship(self, other_memory_id: str) -> None:
        """Add a relationship to another memory"""
        if other_memory_id not in self.relationships:
            self.relationships.append(other_memory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata.to_dict(),
            'relationships': self.relationships,
            'cluster_id': self.cluster_id,
            'importance_score': self.importance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'emotional_context': self.emotional_context,
            'validation_score': self.validation_score
        }


@dataclass 
class MemoryCluster:
    """A cluster of related memories"""
    id: int
    name: str
    description: str
    memory_ids: List[str]
    centroid: List[float]
    importance: float
    created_at: str
    last_updated: str
    
    @classmethod
    def create(cls, cluster_id: int, name: str, description: str,
               memory_ids: List[str], centroid: List[float], 
               importance: float) -> 'MemoryCluster':
        """Factory method to create a new cluster"""
        timestamp = datetime.utcnow().isoformat()
        
        return cls(
            id=cluster_id,
            name=name,
            description=description,
            memory_ids=memory_ids,
            centroid=centroid,
            importance=importance,
            created_at=timestamp,
            last_updated=timestamp
        )
    
    def add_memory(self, memory_id: str) -> None:
        """Add a memory to this cluster"""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.last_updated = datetime.utcnow().isoformat()
    
    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from this cluster"""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            self.last_updated = datetime.utcnow().isoformat()
    
    def update_timestamp(self) -> None:
        """Update the last_updated timestamp"""
        self.last_updated = datetime.utcnow().isoformat()


@dataclass
class MemoryStatistics:
    """Statistics about the memory system"""
    total_memories: int = 0
    total_retrievals: int = 0
    successful_retrievals: int = 0
    failed_retrievals: int = 0
    memory_clusters: int = 0
    avg_cluster_size: float = 0.0
    memory_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_memories': self.total_memories,
            'total_retrievals': self.total_retrievals,
            'successful_retrievals': self.successful_retrievals,
            'failed_retrievals': self.failed_retrievals,
            'memory_clusters': self.memory_clusters,
            'avg_cluster_size': self.avg_cluster_size,
            'memory_relationships': self.memory_relationships
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate retrieval success rate"""
        if self.total_retrievals == 0:
            return 0.0
        return self.successful_retrievals / self.total_retrievals


@dataclass
class RetrievalResult:
    """Result from memory retrieval"""
    memory_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_similarity: float
    importance_score: float
    composite_score: float
    time_decay: float
    access_count: int
    cluster_id: Optional[int]
    relationships: List[str]
    emotional_context: Dict[str, float]
    retrieval_strategy: str
    last_accessed: str
    
    @classmethod
    def from_memory_node(cls, node: MemoryNode, relevance_score: float,
                        semantic_similarity: float, composite_score: float,
                        time_decay: float, retrieval_strategy: str) -> 'RetrievalResult':
        """Create retrieval result from memory node"""
        return cls(
            memory_id=node.id,
            content=node.content,
            metadata=node.metadata.to_dict(),
            relevance_score=relevance_score,
            semantic_similarity=semantic_similarity,
            importance_score=node.importance_score,
            composite_score=composite_score,
            time_decay=time_decay,
            access_count=node.access_count,
            cluster_id=node.cluster_id,
            relationships=node.relationships,
            emotional_context=node.emotional_context,
            retrieval_strategy=retrieval_strategy,
            last_accessed=node.last_accessed
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.memory_id,
            'text': self.content,
            'metadata': self.metadata,
            'relevance': self.relevance_score,
            'semantic_similarity': self.semantic_similarity,
            'importance_score': self.importance_score,
            'composite_score': self.composite_score,
            'time_decay': self.time_decay,
            'access_count': self.access_count,
            'cluster_id': self.cluster_id,
            'relationships': self.relationships,
            'emotional_context': self.emotional_context,
            'retrieval_strategy': self.retrieval_strategy,
            'last_accessed': self.last_accessed
        } 