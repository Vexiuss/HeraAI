import os
import sys
import warnings
import json
import math
import numpy as np
import hashlib
from io import StringIO
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
import contextlib
import uuid
import re
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set environment variables to disable telemetry
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress stderr temporarily during ChromaDB initialization to hide telemetry errors
class SuppressStderr(contextlib.redirect_stderr):
    def __init__(self):
        super().__init__(open(os.devnull, 'w'))

# Import ChromaDB with error suppression
with SuppressStderr():
    import chromadb

from sentence_transformers import SentenceTransformer
import re

# Enhanced Memory Configuration for Long-Term Memory
MEMORY_CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "max_memories_per_user": 50000,  # Much higher limit per user
    "consolidation_threshold": 5000,  # Consolidate after more memories
    "importance_decay_days": 365,  # Memories last much longer (1 year)
    "conversation_session_timeout": 3600,  # 1 hour sessions
    "semantic_similarity_threshold": 0.7,
    "long_term_retention_days": 1095,  # 3 years for long-term memories
    "user_data_dir": "user_data",
    "memory_types": {
        "conversation": {"weight": 1.0, "decay_rate": 0.995, "retention_days": 90},
        "fact": {"weight": 2.0, "decay_rate": 0.999, "retention_days": 1095},  # 3 years
        "preference": {"weight": 1.8, "decay_rate": 0.998, "retention_days": 730},  # 2 years
        "personal": {"weight": 2.5, "decay_rate": 0.9995, "retention_days": 1825},  # 5 years
        "context": {"weight": 0.8, "decay_rate": 0.99, "retention_days": 30},
        "summary": {"weight": 1.5, "decay_rate": 0.997, "retention_days": 365},
        "relationship": {"weight": 2.2, "decay_rate": 0.9998, "retention_days": 1460}  # 4 years
    },
    "user_profile_fields": [
        "name", "preferences", "interests", "important_facts", 
        "relationships", "goals", "personality_traits", "conversation_style"
    ]
}

# Enhanced memory data structures
@dataclass
class MemoryNode:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    relationships: List[str]  # IDs of related memories
    cluster_id: Optional[int] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: str = ""
    emotional_context: Dict[str, float] = None
    validation_score: float = 1.0
    
    def __post_init__(self):
        if self.emotional_context is None:
            self.emotional_context = {}

@dataclass
class MemoryCluster:
    id: int
    name: str
    description: str
    memory_ids: List[str]
    centroid: List[float]
    importance: float
    created_at: str
    last_updated: str

class AdvancedMemorySystem:
    def __init__(self):
        self.model = None
        self.client = None
        self.collection = None
        self.current_user = None
        self.current_user_id = None
        self.memory_nodes = {}  # Cache for memory nodes
        self.clusters = {}  # Memory clusters
        self.relationship_graph = defaultdict(set)  # Memory relationships
        self.memory_stats = {
            'total_memories': 0,
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'memory_clusters': 0,
            'avg_cluster_size': 0,
            'memory_relationships': 0
        }
        self.load_model()
        self.init_client()
        
    def load_model(self):
        """Load the sentence transformer model"""
        print("Loading advanced embedding model for enhanced memory...")
        try:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            print("✅ Advanced embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to smaller model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Fallback model loaded")
    
    def init_client(self):
        """Initialize ChromaDB client"""
        with SuppressStderr():
            self.client = chromadb.PersistentClient(
                path="./memory_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
    
    def parse_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON strings in metadata back to dictionaries"""
        parsed_metadata = metadata.copy()
        
        # Parse emotional_context if it's a JSON string
        if 'emotional_context' in parsed_metadata and isinstance(parsed_metadata['emotional_context'], str):
            try:
                parsed_metadata['emotional_context'] = json.loads(parsed_metadata['emotional_context'])
            except (json.JSONDecodeError, TypeError):
                parsed_metadata['emotional_context'] = {}
        
        # Parse context if it's a JSON string
        if 'context' in parsed_metadata and isinstance(parsed_metadata['context'], str):
            try:
                parsed_metadata['context'] = json.loads(parsed_metadata['context'])
            except (json.JSONDecodeError, TypeError):
                parsed_metadata['context'] = {}
        
        return parsed_metadata
    
    def set_user(self, username: str) -> str:
        """Set current user and create/load their memory collection"""
        self.current_user = username
        self.current_user_id = hashlib.md5(username.encode()).hexdigest()[:16]
        collection_name = f"user_memory_{self.current_user_id}"
        
        with SuppressStderr():
            try:
                self.collection = self.client.get_collection(name=collection_name)
                print(f"📚 Loaded existing memory collection for {username}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"user": username, "created_at": datetime.utcnow().isoformat()}
                )
                print(f"🆕 Created new memory collection for {username}")
        
        # Load user profile and memory structures
        self.load_user_data()
        self.update_memory_stats()
        return self.current_user_id
    
    def load_user_data(self):
        """Load user-specific memory data structures"""
        user_dir = f"user_data"
        os.makedirs(user_dir, exist_ok=True)
        
        # Load memory nodes
        nodes_file = f"{user_dir}/memory_nodes_{self.current_user_id}.pkl"
        if os.path.exists(nodes_file):
            try:
                with open(nodes_file, 'rb') as f:
                    self.memory_nodes = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Error loading memory nodes: {e}")
                self.memory_nodes = {}
        
        # Load clusters
        clusters_file = f"{user_dir}/memory_clusters_{self.current_user_id}.pkl"
        if os.path.exists(clusters_file):
            try:
                with open(clusters_file, 'rb') as f:
                    self.clusters = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Error loading memory clusters: {e}")
                self.clusters = {}
        
        # Load relationship graph
        relationships_file = f"{user_dir}/memory_relationships_{self.current_user_id}.pkl"
        if os.path.exists(relationships_file):
            try:
                with open(relationships_file, 'rb') as f:
                    self.relationship_graph = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Error loading memory relationships: {e}")
                self.relationship_graph = defaultdict(set)
    
    def save_user_data(self):
        """Save user-specific memory data structures"""
        user_dir = f"user_data"
        os.makedirs(user_dir, exist_ok=True)
        
        # Save memory nodes
        try:
            with open(f"{user_dir}/memory_nodes_{self.current_user_id}.pkl", 'wb') as f:
                pickle.dump(self.memory_nodes, f)
        except Exception as e:
            print(f"⚠️ Error saving memory nodes: {e}")
        
        # Save clusters
        try:
            with open(f"{user_dir}/memory_clusters_{self.current_user_id}.pkl", 'wb') as f:
                pickle.dump(self.clusters, f)
        except Exception as e:
            print(f"⚠️ Error saving memory clusters: {e}")
        
        # Save relationship graph
        try:
            with open(f"{user_dir}/memory_relationships_{self.current_user_id}.pkl", 'wb') as f:
                pickle.dump(dict(self.relationship_graph), f)
        except Exception as e:
            print(f"⚠️ Error saving memory relationships: {e}")
    
    def analyze_emotional_context(self, text: str) -> Dict[str, float]:
        """Analyze emotional context of memory content"""
        emotion_keywords = {
            'joy': ['happy', 'excited', 'glad', 'pleased', 'delighted', 'thrilled', 'wonderful', 'amazing'],
            'sadness': ['sad', 'disappointed', 'upset', 'down', 'depressed', 'unhappy', 'terrible', 'awful'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'outraged'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned', 'frightened'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'sudden'],
            'curiosity': ['curious', 'interested', 'wondering', 'intrigued', 'fascinated'],
            'achievement': ['accomplished', 'successful', 'proud', 'achieved', 'completed', 'finished']
        }
        
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions[emotion] = min(score / len(keywords), 1.0)
        
        return emotions
    
    def calculate_advanced_importance(self, text: str, role: str, mem_type: str, 
                                    emotional_context: Dict[str, float]) -> float:
        """Calculate advanced importance score with multiple factors"""
        base_score = 0.3
        
        # Role-based importance
        role_weights = {"user": 0.7, "ai": 0.3}
        role_score = role_weights.get(role, 0.5)
        
        # Memory type importance
        type_weights = {
            "personal": 0.9, "fact": 0.8, "preference": 0.7, 
            "context": 0.6, "summary": 0.8, "conversation": 0.4
        }
        type_score = type_weights.get(mem_type, 0.5)
        
        # Content-based importance indicators
        importance_indicators = [
            'important', 'remember', 'never forget', 'always', 'crucial', 'essential',
            'my name is', 'i am', 'i love', 'i hate', 'my goal', 'my dream',
            'my family', 'my job', 'my birthday', 'my phone', 'my address'
        ]
        
        text_lower = text.lower()
        content_score = 0.0
        for indicator in importance_indicators:
            if indicator in text_lower:
                content_score += 0.1
        content_score = min(content_score, 0.5)
        
        # Emotional importance
        emotional_score = 0.0
        if emotional_context:
            high_emotion_weights = {'joy': 0.8, 'sadness': 0.9, 'anger': 0.7, 'achievement': 0.9}
            for emotion, intensity in emotional_context.items():
                if emotion in high_emotion_weights:
                    emotional_score += intensity * high_emotion_weights[emotion]
        emotional_score = min(emotional_score, 0.3)
        
        # Length-based importance (longer memories often more important)
        length_score = min(len(text) / 1000, 0.2)
        
        # Combine all factors
        total_score = base_score + role_score + type_score + content_score + emotional_score + length_score
        return min(total_score, 1.0)
    
    def find_memory_relationships(self, new_memory_id: str, new_embedding: List[float], 
                                threshold: float = 0.7) -> List[str]:
        """Find relationships between memories based on semantic similarity"""
        related_memories = []
        
        for memory_id, node in self.memory_nodes.items():
            if memory_id != new_memory_id:
                similarity = cosine_similarity([new_embedding], [node.embedding])[0][0]
                if similarity > threshold:
                    related_memories.append(memory_id)
                    # Add bidirectional relationship
                    self.relationship_graph[new_memory_id].add(memory_id)
                    self.relationship_graph[memory_id].add(new_memory_id)
        
        return related_memories
    
    def save_memory(self, text: str, role: str, mem_type: str = "conversation", 
                   session_id: str = None, context: Dict[str, Any] = None) -> str:
        """Save memory with advanced features"""
        if not self.collection:
            print("❌ No user set. Please set a user first.")
            return None
        
        # Generate embedding
        embedding = self.model.encode(text).tolist()
        
        # Analyze emotional context
        emotional_context = self.analyze_emotional_context(text)
        
        # Calculate importance
        importance = self.calculate_advanced_importance(text, role, mem_type, emotional_context)
        
        # Create memory node
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        metadata = {
            "role": role,
            "type": mem_type,
            "timestamp": timestamp,
            "session_id": session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "importance": importance,
            "user_id": self.current_user_id,
            "user_name": self.current_user,
            "emotional_context": json.dumps(emotional_context) if emotional_context else "{}",
            "context": json.dumps(context) if context else "{}"
        }
        
        # Find relationships
        relationships = self.find_memory_relationships(memory_id, embedding)
        
        # Create memory node
        memory_node = MemoryNode(
            id=memory_id,
            content=text,
            embedding=embedding,
            metadata=metadata,
            relationships=relationships,
            importance_score=importance,
            access_count=0,
            last_accessed=timestamp,
            emotional_context=emotional_context
        )
        
        # Save to ChromaDB
        with SuppressStderr():
            try:
                self.collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[memory_id]
                )
                
                # Add to memory nodes cache
                self.memory_nodes[memory_id] = memory_node
                
                # Update stats
                self.memory_stats['total_memories'] += 1
                self.memory_stats['memory_relationships'] += len(relationships)
                
                # Trigger clustering if we have enough memories
                if len(self.memory_nodes) % 10 == 0:  # Cluster every 10 memories
                    self.update_memory_clusters()
                
                # Save user data
                self.save_user_data()
                
                return memory_id
                
            except Exception as e:
                print(f"❌ Error saving memory: {e}")
                return None
    
    def update_memory_clusters(self):
        """Update memory clusters using K-means clustering"""
        if len(self.memory_nodes) < 3:
            return
        
        try:
            # Prepare embeddings and metadata
            embeddings = []
            memory_ids = []
            
            for memory_id, node in self.memory_nodes.items():
                embeddings.append(node.embedding)
                memory_ids.append(memory_id)
            
            # Determine optimal number of clusters
            n_memories = len(embeddings)
            n_clusters = min(max(2, n_memories // 5), 10)  # 2-10 clusters
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Update memory nodes with cluster info
            for i, memory_id in enumerate(memory_ids):
                self.memory_nodes[memory_id].cluster_id = int(cluster_labels[i])
            
            # Create cluster objects
            new_clusters = {}
            for cluster_id in range(n_clusters):
                cluster_memories = [memory_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_memories:
                    # Generate cluster name and description
                    cluster_texts = [self.memory_nodes[mid].content for mid in cluster_memories[:5]]
                    cluster_name = self.generate_cluster_name(cluster_texts)
                    cluster_description = self.generate_cluster_description(cluster_texts)
                    
                    # Calculate cluster importance
                    cluster_importance = np.mean([self.memory_nodes[mid].importance_score for mid in cluster_memories])
                    
                    new_clusters[cluster_id] = MemoryCluster(
                        id=cluster_id,
                        name=cluster_name,
                        description=cluster_description,
                        memory_ids=cluster_memories,
                        centroid=kmeans.cluster_centers_[cluster_id].tolist(),
                        importance=cluster_importance,
                        created_at=datetime.utcnow().isoformat(),
                        last_updated=datetime.utcnow().isoformat()
                    )
            
            self.clusters = new_clusters
            self.memory_stats['memory_clusters'] = len(self.clusters)
            self.memory_stats['avg_cluster_size'] = np.mean([len(cluster.memory_ids) for cluster in self.clusters.values()])
            
            print(f"🔄 Updated memory clusters: {len(self.clusters)} clusters created")
            
        except Exception as e:
            print(f"⚠️ Error updating clusters: {e}")
    
    def generate_cluster_name(self, texts: List[str]) -> str:
        """Generate a name for a memory cluster"""
        # Extract key words from texts
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend([w for w in words if len(w) > 3])
        
        # Find most common meaningful words
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(3) if count > 1]
        
        if common_words:
            return f"Cluster: {', '.join(common_words[:2])}"
        else:
            return "Mixed Topics"
    
    def generate_cluster_description(self, texts: List[str]) -> str:
        """Generate a description for a memory cluster"""
        # Simple description based on content analysis
        if any('personal' in text.lower() for text in texts):
            return "Personal information and preferences"
        elif any(word in ' '.join(texts).lower() for word in ['goal', 'plan', 'want', 'dream']):
            return "Goals and aspirations"
        elif any(word in ' '.join(texts).lower() for word in ['work', 'job', 'career']):
            return "Professional and career-related"
        elif any(word in ' '.join(texts).lower() for word in ['family', 'friend', 'relationship']):
            return "Social relationships and connections"
        else:
            return "General conversation topics"
    
    def retrieve_memories_ultra_advanced(self, query: str, top_k: int = 10, 
                                       include_clusters: bool = True,
                                       include_relationships: bool = True) -> List[Dict[str, Any]]:
        """Ultra-advanced memory retrieval with multiple strategies"""
        if not self.collection:
            return []
        
        try:
            # Update access statistics
            self.memory_stats['total_retrievals'] += 1
            
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Strategy 1: Direct semantic search
            with SuppressStderr():
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,  # Get more results for filtering
                    include=['documents', 'metadatas', 'distances']
                )
            
            # Strategy 2: Cluster-based retrieval
            cluster_results = []
            if include_clusters and self.clusters:
                for cluster in self.clusters.values():
                    cluster_similarity = cosine_similarity([query_embedding], [cluster.centroid])[0][0]
                    if cluster_similarity > 0.3:  # Threshold for cluster relevance
                        cluster_results.extend(cluster.memory_ids[:3])  # Top 3 from each relevant cluster
            
            # Strategy 3: Relationship-based retrieval
            relationship_results = []
            if include_relationships:
                # Find memories related to top semantic matches
                for memory_id in results['ids'][0][:5]:
                    if memory_id in self.relationship_graph:
                        relationship_results.extend(list(self.relationship_graph[memory_id])[:2])
            
            # Combine and deduplicate results
            all_memory_ids = list(set(results['ids'][0] + cluster_results + relationship_results))
            
            # Prepare enhanced results
            enhanced_results = []
            for i, memory_id in enumerate(all_memory_ids[:top_k]):
                if memory_id in self.memory_nodes:
                    node = self.memory_nodes[memory_id]
                    
                    # Update access count
                    node.access_count += 1
                    node.last_accessed = datetime.utcnow().isoformat()
                    
                    # Calculate comprehensive score
                    semantic_similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
                    
                    # Time decay factor
                    memory_time = datetime.fromisoformat(node.metadata['timestamp'])
                    time_diff = datetime.utcnow() - memory_time
                    time_decay = max(0.1, 1.0 - (time_diff.days / 365.0))  # Decay over a year
                    
                    # Access frequency boost
                    access_boost = min(0.3, node.access_count * 0.05)
                    
                    # Relationship boost
                    relationship_boost = 0.1 if memory_id in relationship_results else 0.0
                    
                    # Cluster boost
                    cluster_boost = 0.1 if memory_id in cluster_results else 0.0
                    
                    # Composite score
                    composite_score = (
                        semantic_similarity * 0.4 +
                        node.importance_score * 0.3 +
                        time_decay * 0.1 +
                        access_boost +
                        relationship_boost +
                        cluster_boost
                    )
                    
                    result = {
                        'id': memory_id,
                        'text': node.content,
                        'metadata': node.metadata,
                        'semantic_similarity': semantic_similarity,
                        'importance_score': node.importance_score,
                        'composite_score': composite_score,
                        'time_decay': time_decay,
                        'access_count': node.access_count,
                        'cluster_id': node.cluster_id,
                        'relationships': node.relationships,
                        'emotional_context': node.emotional_context,
                        'retrieval_strategy': self.get_retrieval_strategy(memory_id, results['ids'][0], 
                                                                        cluster_results, relationship_results)
                    }
                    
                    enhanced_results.append(result)
            
            # Sort by composite score
            enhanced_results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Apply quality filters
            filtered_results = []
            for result in enhanced_results:
                # Quality thresholds
                if (result['semantic_similarity'] > 0.2 or 
                    result['importance_score'] > 0.6 or
                    result['access_count'] > 2):
                    filtered_results.append(result)
            
            if filtered_results:
                self.memory_stats['successful_retrievals'] += 1
            else:
                self.memory_stats['failed_retrievals'] += 1
            
            return filtered_results[:top_k]
            
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            self.memory_stats['failed_retrievals'] += 1
            return []
    
    def get_retrieval_strategy(self, memory_id: str, semantic_ids: List[str], 
                             cluster_ids: List[str], relationship_ids: List[str]) -> str:
        """Determine which retrieval strategy found this memory"""
        strategies = []
        if memory_id in semantic_ids:
            strategies.append("semantic")
        if memory_id in cluster_ids:
            strategies.append("cluster")
        if memory_id in relationship_ids:
            strategies.append("relationship")
        return "+".join(strategies) if strategies else "unknown"
    
    def validate_memory_consistency(self) -> Dict[str, Any]:
        """Validate memory consistency and detect conflicts"""
        validation_results = {
            'total_memories': len(self.memory_nodes),
            'conflicts_detected': 0,
            'inconsistencies': [],
            'duplicate_candidates': [],
            'validation_score': 1.0
        }
        
        try:
            # Check for potential duplicates
            embeddings = []
            memory_ids = []
            
            for memory_id, node in self.memory_nodes.items():
                embeddings.append(node.embedding)
                memory_ids.append(memory_id)
            
            if len(embeddings) > 1:
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        if similarity_matrix[i][j] > 0.95:  # Very high similarity
                            validation_results['duplicate_candidates'].append({
                                'memory1': memory_ids[i],
                                'memory2': memory_ids[j],
                                'similarity': similarity_matrix[i][j],
                                'content1': self.memory_nodes[memory_ids[i]].content[:100],
                                'content2': self.memory_nodes[memory_ids[j]].content[:100]
                            })
            
            # Check for temporal inconsistencies
            user_facts = {}
            for memory_id, node in self.memory_nodes.items():
                if node.metadata.get('type') == 'personal' or node.metadata.get('type') == 'fact':
                    content_lower = node.content.lower()
                    
                    # Extract facts about user
                    if 'my name is' in content_lower:
                        name = re.search(r'my name is (\w+)', content_lower)
                        if name:
                            fact_key = 'name'
                            fact_value = name.group(1)
                            if fact_key in user_facts and user_facts[fact_key] != fact_value:
                                validation_results['inconsistencies'].append({
                                    'type': 'conflicting_facts',
                                    'fact': fact_key,
                                    'values': [user_facts[fact_key], fact_value],
                                    'memory_ids': [user_facts[f'{fact_key}_memory'], memory_id]
                                })
                            else:
                                user_facts[fact_key] = fact_value
                                user_facts[f'{fact_key}_memory'] = memory_id
            
            validation_results['conflicts_detected'] = len(validation_results['inconsistencies'])
            
            # Calculate overall validation score
            total_issues = len(validation_results['duplicate_candidates']) + len(validation_results['inconsistencies'])
            if validation_results['total_memories'] > 0:
                validation_results['validation_score'] = max(0.0, 1.0 - (total_issues / validation_results['total_memories']))
            
        except Exception as e:
            print(f"⚠️ Error during memory validation: {e}")
        
        return validation_results
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """Get comprehensive memory analytics"""
        analytics = {
            'basic_stats': self.memory_stats.copy(),
            'memory_distribution': {},
            'temporal_patterns': {},
            'emotional_patterns': {},
            'cluster_analysis': {},
            'relationship_analysis': {},
            'user_behavior': {}
        }
        
        try:
            # Memory type distribution
            type_counts = defaultdict(int)
            role_counts = defaultdict(int)
            daily_counts = defaultdict(int)
            emotion_counts = defaultdict(int)
            
            for node in self.memory_nodes.values():
                # Type distribution
                mem_type = node.metadata.get('type', 'unknown')
                type_counts[mem_type] += 1
                
                # Role distribution
                role = node.metadata.get('role', 'unknown')
                role_counts[role] += 1
                
                # Daily patterns
                timestamp = node.metadata.get('timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp).date()
                        daily_counts[str(date)] += 1
                    except:
                        pass
                
                # Emotional patterns
                for emotion in node.emotional_context:
                    emotion_counts[emotion] += 1
            
            analytics['memory_distribution'] = {
                'by_type': dict(type_counts),
                'by_role': dict(role_counts)
            }
            
            analytics['temporal_patterns'] = {
                'daily_counts': dict(daily_counts),
                'most_active_day': max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
            }
            
            analytics['emotional_patterns'] = dict(emotion_counts)
            
            # Cluster analysis
            if self.clusters:
                cluster_sizes = [len(cluster.memory_ids) for cluster in self.clusters.values()]
                cluster_importance = [cluster.importance for cluster in self.clusters.values()]
                
                analytics['cluster_analysis'] = {
                    'total_clusters': len(self.clusters),
                    'avg_cluster_size': np.mean(cluster_sizes),
                    'largest_cluster': max(cluster_sizes),
                    'avg_cluster_importance': np.mean(cluster_importance),
                    'cluster_names': [cluster.name for cluster in self.clusters.values()]
                }
            
            # Relationship analysis
            if self.relationship_graph:
                relationship_counts = [len(related) for related in self.relationship_graph.values()]
                analytics['relationship_analysis'] = {
                    'total_relationships': sum(relationship_counts),
                    'avg_relationships_per_memory': np.mean(relationship_counts) if relationship_counts else 0,
                    'most_connected_memory': max(self.relationship_graph.items(), 
                                                key=lambda x: len(x[1]))[0] if relationship_counts else None
                }
            
            # User behavior analysis
            access_counts = [node.access_count for node in self.memory_nodes.values()]
            importance_scores = [node.importance_score for node in self.memory_nodes.values()]
            
            analytics['user_behavior'] = {
                'avg_access_count': np.mean(access_counts) if access_counts else 0,
                'most_accessed_memory': max(self.memory_nodes.items(), 
                                          key=lambda x: x[1].access_count)[0] if access_counts else None,
                'avg_importance_score': np.mean(importance_scores) if importance_scores else 0,
                'high_importance_memories': len([s for s in importance_scores if s > 0.7])
            }
            
        except Exception as e:
            print(f"⚠️ Error generating analytics: {e}")
        
        return analytics
    
    def export_memories(self, format: str = 'json') -> str:
        """Export memories in specified format"""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"memory_export_{self.current_user_id}_{timestamp}.{format}"
            
            export_data = {
                'user_id': self.current_user_id,
                'user_name': self.current_user,
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_memories': len(self.memory_nodes),
                'memories': [],
                'clusters': {},
                'relationships': dict(self.relationship_graph),
                'analytics': self.get_memory_analytics()
            }
            
            # Export memory nodes (without embeddings for size)
            for memory_id, node in self.memory_nodes.items():
                memory_data = {
                    'id': node.id,
                    'content': node.content,
                    'metadata': node.metadata,
                    'importance_score': node.importance_score,
                    'access_count': node.access_count,
                    'last_accessed': node.last_accessed,
                    'emotional_context': node.emotional_context,
                    'relationships': node.relationships,
                    'cluster_id': node.cluster_id
                }
                export_data['memories'].append(memory_data)
            
            # Export clusters
            for cluster_id, cluster in self.clusters.items():
                export_data['clusters'][str(cluster_id)] = asdict(cluster)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Memories exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error exporting memories: {e}")
            return None
    
    def smart_memory_compression(self, days_threshold: int = 30) -> int:
        """Compress old memories while preserving important information"""
        compressed_count = 0
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # Group old memories by similarity and importance
            old_memories = []
            for memory_id, node in self.memory_nodes.items():
                memory_time = datetime.fromisoformat(node.metadata['timestamp'])
                if memory_time < cutoff_date and node.importance_score < 0.6:
                    old_memories.append((memory_id, node))
            
            if len(old_memories) < 5:  # Not enough memories to compress
                return 0
            
            # Cluster similar old memories
            embeddings = [node.embedding for _, node in old_memories]
            memory_ids = [memory_id for memory_id, _ in old_memories]
            
            n_clusters = max(1, len(old_memories) // 10)  # Compress to 1/10th
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Create compressed summaries
            for cluster_id in range(n_clusters):
                cluster_memories = [old_memories[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_memories) > 2:  # Only compress if multiple memories
                    # Create summary
                    contents = [node.content for _, node in cluster_memories]
                    summary = f"Summary of {len(contents)} conversations about similar topics: " + \
                             "; ".join(contents[:3])  # First 3 as summary
                    
                    # Calculate average importance
                    avg_importance = np.mean([node.importance_score for _, node in cluster_memories])
                    
                    # Create compressed memory
                    compressed_id = str(uuid.uuid4())
                    compressed_embedding = np.mean([node.embedding for _, node in cluster_memories], axis=0).tolist()
                    
                    compressed_metadata = {
                        'role': 'system',
                        'type': 'summary',
                        'timestamp': datetime.utcnow().isoformat(),
                        'importance': avg_importance,
                        'user_id': self.current_user_id,
                        'user_name': self.current_user,
                        'compressed_from': len(cluster_memories),
                        'original_timespan': f"{cluster_memories[0][1].metadata['timestamp']} to {cluster_memories[-1][1].metadata['timestamp']}"
                    }
                    
                    # Save compressed memory
                    with SuppressStderr():
                        self.collection.add(
                            documents=[summary],
                            embeddings=[compressed_embedding],
                            metadatas=[compressed_metadata],
                            ids=[compressed_id]
                        )
                    
                    # Remove original memories
                    for memory_id, _ in cluster_memories:
                        with SuppressStderr():
                            self.collection.delete(ids=[memory_id])
                        if memory_id in self.memory_nodes:
                            del self.memory_nodes[memory_id]
                        compressed_count += 1
            
            print(f"🗜️ Compressed {compressed_count} old memories into summaries")
            
        except Exception as e:
            print(f"⚠️ Error during memory compression: {e}")
        
        return compressed_count
    
    def update_memory_stats(self):
        """Update memory statistics"""
        try:
            if self.collection:
                with SuppressStderr():
                    count_result = self.collection.count()
                    self.memory_stats['total_memories'] = count_result
        except Exception as e:
            print(f"⚠️ Error updating memory stats: {e}")

# Global instance
memory_system = AdvancedMemorySystem()

# Updated API functions
def set_user(user_name: str) -> str:
    """Set the current user for memory operations"""
    return memory_system.set_user(user_name)

def save_memory(text: str, role: str, mem_type: str = "conversation") -> str:
    """Save memory for current user"""
    return memory_system.save_memory(text, role, mem_type)

def retrieve_memories(query: str, top_k: int = 10, role: str = None, mem_type: str = None) -> List[str]:
    """Retrieve memories for current user - backward compatible"""
    try:
        memories = memory_system.retrieve_memories_ultra_advanced(query, top_k)
        
        # Filter by role and type if specified
        filtered_memories = []
        for mem in memories:
            if role and mem['metadata'].get('role') != role:
                continue
            if mem_type and mem['metadata'].get('type') != mem_type:
                continue
            filtered_memories.append(mem)
        
        return [mem["text"] for mem in filtered_memories[:top_k]]
    except Exception as e:
        print(f"⚠️ Error retrieving memories: {e}")
        return []

def get_conversation_context(query: str, max_length: int = 2500) -> str:
    """Get intelligent conversation context for current user"""
    try:
        # Use advanced retrieval to get relevant memories
        memories = memory_system.retrieve_memories_ultra_advanced(query, top_k=15)
        
        if not memories:
            return ""
        
        # Build context string with quality filtering
        context_lines = []
        total_length = 0
        
        for memory in memories:
            if memory['composite_score'] > 0.3:  # Quality threshold
                role = memory['metadata'].get('role', 'user')
                mem_type = memory['metadata'].get('type', 'conversation')
                content = memory['text']
                
                # Format based on memory type
                if mem_type in ['personal', 'fact', 'preference']:
                    line = f"[{mem_type.upper()}] {content}"
                else:
                    line = f"[{role.upper()}]: {content}"
                
                # Check length limit
                if total_length + len(line) > max_length:
                    break
                
                context_lines.append(line)
                total_length += len(line)
        
        return "\n".join(context_lines) if context_lines else ""
        
    except Exception as e:
        print(f"⚠️ Error building conversation context: {e}")
        return ""

def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics for current user"""
    return memory_system.memory_stats

def retrieve_memories_advanced(query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Advanced retrieve function with full metadata for current user"""
    return memory_system.retrieve_memories_ultra_advanced(query, top_k, **kwargs)

def get_user_profile() -> Dict[str, Any]:
    """Get current user's profile"""
    # Implementation needed
    return {}

def update_user_profile(updates: Dict[str, Any]):
    """Update current user's profile"""
    # Implementation needed
    return False

def list_users() -> List[Dict[str, str]]:
    """List all users"""
    # Implementation needed
    return []
