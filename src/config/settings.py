"""
Configuration settings for HeraAI

This module contains all configuration constants organized by functionality.
All settings should be modified here rather than scattered throughout the codebase.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent.parent

# Environment variables for telemetry suppression
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# ================================
# MEMORY SYSTEM CONFIGURATION
# ================================
MEMORY_CONFIG: Dict[str, Any] = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "fallback_model": "all-MiniLM-L6-v2",
    "max_memories_per_user": 50000,
    "consolidation_threshold": 5000,
    "importance_decay_days": 365,
    "conversation_session_timeout": 3600,  # 1 hour
    "semantic_similarity_threshold": 0.7,
    "long_term_retention_days": 1095,  # 3 years
    "user_data_dir": "user_data",
    "memory_db_path": "./memory_db",
    
    # Anti-hallucination thresholds
    "strict_semantic_threshold": 0.4,
    "min_importance_for_weak_matches": 0.85,
    "access_count_threshold": 3,
    "min_access_semantic_threshold": 0.3,
    
    # Memory type configurations
    "memory_types": {
        "conversation": {"weight": 1.0, "decay_rate": 0.995, "retention_days": 90},
        "fact": {"weight": 2.0, "decay_rate": 0.999, "retention_days": 1095},
        "preference": {"weight": 1.8, "decay_rate": 0.998, "retention_days": 730},
        "personal": {"weight": 2.5, "decay_rate": 0.9995, "retention_days": 1825},
        "context": {"weight": 0.8, "decay_rate": 0.99, "retention_days": 30},
        "summary": {"weight": 1.5, "decay_rate": 0.997, "retention_days": 365},
        "relationship": {"weight": 2.2, "decay_rate": 0.9998, "retention_days": 1460}
    },
    
    # User profile fields
    "user_profile_fields": [
        "name", "preferences", "interests", "important_facts",
        "relationships", "goals", "personality_traits", "conversation_style"
    ],
    
    # Clustering settings
    "clustering": {
        "min_memories_for_clustering": 3,
        "max_clusters": 10,
        "memories_per_cluster_ratio": 5,  # memories // 5 = clusters
        "cluster_update_frequency": 10,  # every 10 new memories
    }
}

# ================================
# AUDIO SYSTEM CONFIGURATION  
# ================================
AUDIO_CONFIG: Dict[str, Any] = {
    # Speech Recognition
    "speech_recognition": {
        "pause_threshold": 1.0,
        "energy_threshold": 300,
        "dynamic_energy_threshold": True,
        "dynamic_energy_adjustment_damping": 0.15,
        "dynamic_energy_ratio": 1.5,
        "phrase_time_limit": 30,
        "non_speaking_duration": 0.5,
        "ambient_noise_duration": 0.8,
        "timeout": 2,
    },
    
    # Voice Interruption Detection
    "interruption": {
        "pause_threshold": 0.5,
        "energy_threshold": 150,
        "non_speaking_duration": 0.2,
        "detection_timeout": 0.05,
        "phrase_time_limit": 0.5,
        "ambient_noise_duration": 0.1,
    },
    
    # Text-to-Speech
    "tts": {
        "voice": "en-US-JennyNeural",
        "output_file": "output.mp3",
        "audio_check_interval": 0.05,
        "interrupt_check_interval": 0.1,
        "audio_start_timeout": 2,
    }
}

# ================================
# AI MODEL CONFIGURATION
# ================================
AI_CONFIG: Dict[str, Any] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "api_endpoint": "/api/generate",
        "default_model": "llama3",
        "stream": False,
        "timeout": 30,
    },
    
    # Conversation settings
    "conversation": {
        "max_context_length": 2500,
        "recent_messages_count": 8,
        "max_background_lines": 8,
        "min_line_length": 15,
    },
    
    # System prompts
    "prompts": {
        "base_system_prompt": """You are Hera, an advanced AI assistant with long-term memory capabilities. 
        You maintain detailed memories of conversations, user preferences, and personal information. 
        You are helpful, conversational, and personalized based on what you remember about the user.
        
        Important: Only reference information that is explicitly provided in the background context. 
        Do not make assumptions or create information that wasn't shared with you.""",
        
        "no_memory_prompt": """You are Hera, an advanced AI assistant. This appears to be a new conversation 
        with limited context. Be helpful and conversational while asking questions to learn about the user."""
    }
}

# ================================
# SYSTEM CONFIGURATION
# ================================
SYSTEM_CONFIG: Dict[str, Any] = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "heraai.log",
    },
    
    "files": {
        "requirements": "requirements.txt",
        "readme": "README.md",
        "license": "LICENSE",
    },
    
    "directories": {
        "user_data": "user_data",
        "memory_db": "memory_db",
        "logs": "logs",
        "temp": "temp",
    },
    
    # Development settings
    "debug": False,
    "verbose": True,
} 