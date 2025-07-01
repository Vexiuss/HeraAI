"""
ChromaDB Storage Module for HeraAI

Handles all ChromaDB operations for persistent memory storage.
"""

# CRITICAL: Load NumPy 2.0 compatibility first
try:
    from ..utils.numpy_compat import setup_numpy_compatibility
    setup_numpy_compatibility()
except Exception as e:
    print(f"Warning: NumPy compatibility setup failed in storage module: {e}")

import contextlib
import os
import sys
import warnings
from typing import List, Dict, Any, Optional

# Completely disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_SERVER_AUTH_CREDENTIALS_FILE'] = ''
os.environ['CHROMA_SERVER_AUTH_CREDENTIALS'] = ''

# Filter out specific ChromaDB telemetry warnings
warnings.filterwarnings("ignore", message=".*capture.*takes.*positional argument.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*", category=UserWarning)

# Comprehensive suppression class
class SuppressOutput(contextlib.redirect_stderr):
    def __init__(self, suppress_stdout=False):
        self.suppress_stdout = suppress_stdout
        self.devnull = open(os.devnull, 'w')
        super().__init__(self.devnull)
        
    def __enter__(self):
        self.old_stdout = None
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.devnull
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_stdout is not None:
            sys.stdout = self.old_stdout
        result = super().__exit__(exc_type, exc_val, exc_tb)
        self.devnull.close()
        return result

# Import ChromaDB with comprehensive suppressions
with SuppressOutput(suppress_stdout=True):
    import chromadb
    from chromadb.config import Settings

from .models import MemoryNode


class ChromaDBStorage:
    """Handles ChromaDB operations for memory storage"""
    
    def __init__(self, db_path: str):
        """
        Initialize ChromaDB storage
        
        Args:
            db_path: Path to the ChromaDB database
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the ChromaDB client with fallback handling"""
        try:
            with SuppressOutput(suppress_stdout=True):
                # Create settings with all telemetry disabled
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=self.db_path
                )
                
                # Additional environment settings to disable telemetry
                os.environ['ANONYMIZED_TELEMETRY'] = 'False'
                
                self.client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=settings
                )
        except ImportError as e:
            if 'hnswlib' in str(e):
                print("⚠️  ChromaDB requires 'hnswlib' for vector operations")
                print("   Running in limited mode without persistent storage")
                print("   Install hnswlib with: pip install hnswlib")
                self.client = None
            else:
                raise
        except Exception as e:
            print(f"⚠️  ChromaDB initialization failed: {e}")
            print("   Running in limited mode without persistent storage")
            self.client = None
    
    def set_collection(self, collection_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Set or create a collection for the current user
        
        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
        """
        if not self.client:
            print(f"⚠️  Running without persistent storage - collection {collection_name} not created")
            self.collection = None
            return
            
        with SuppressOutput():
            try:
                self.collection = self.client.get_collection(name=collection_name)
                print(f"📚 Loaded existing memory collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata=metadata or {}
                )
                print(f"🆕 Created new memory collection: {collection_name}")
    
    def save_memory(self, memory_node: MemoryNode) -> bool:
        """
        Save a memory node to ChromaDB
        
        Args:
            memory_node: The memory node to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            print("⚠️  Persistent storage unavailable - memory not saved to ChromaDB")
            return False
            
        if not self.collection:
            print("❌ No collection set for storage")
            return False
        
        try:
            # Prepare metadata for ChromaDB (must be JSON serializable)
            metadata = memory_node.metadata.to_dict()
            
            # Add the memory to ChromaDB with telemetry suppressed
            with SuppressOutput():
                self.collection.add(
                    embeddings=[memory_node.embedding],
                    documents=[memory_node.content],
                    metadatas=[metadata],
                    ids=[memory_node.id]
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving memory to storage: {e}")
            return False
    
    def search_memories(self, query_embedding: List[float], n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with metadata
        """
        if not self.client:
            print("⚠️  Persistent storage unavailable - no search results")
            return []
            
        if not self.collection:
            print("❌ No collection set for storage")
            return []
        
        try:
            # Query ChromaDB with telemetry suppressed
            with SuppressOutput():
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i, memory_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        'id': memory_id,
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching memories: {e}")
            return []
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID
        
        Args:
            memory_id: The memory ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Memory data or None if not found
        """
        if not self.collection:
            return None
        
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=['documents', 'metadatas']
            )
            
            if result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting memory {memory_id}: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: The memory ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.collection:
            return False
        
        try:
            self.collection.delete(ids=[memory_id])
            return True
            
        except Exception as e:
            print(f"❌ Error deleting memory {memory_id}: {e}")
            return False
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories from the current collection
        
        Returns:
            List[Dict[str, Any]]: All memories with metadata
        """
        if not self.collection:
            return []
        
        try:
            result = self.collection.get(include=['documents', 'metadatas'])
            
            memories = []
            if result['ids']:
                for i, memory_id in enumerate(result['ids']):
                    memories.append({
                        'id': memory_id,
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i]
                    })
            
            return memories
            
        except Exception as e:
            print(f"❌ Error getting all memories: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_memories': count,
                'collection_name': self.collection.name
            }
            
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all memories from the current collection
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.collection:
            return False
        
        try:
            # Get all IDs and delete them
            result = self.collection.get()
            if result['ids']:
                self.collection.delete(ids=result['ids'])
            return True
            
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False 