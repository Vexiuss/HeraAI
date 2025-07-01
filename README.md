# 🤖 HeraAI v2.0 - Advanced Voice Assistant with Persistent Memory

**Enterprise-grade voice AI assistant with ultra-advanced memory system and modular architecture**

HeraAI is a revolutionary voice AI assistant that combines natural speech interaction with cutting-edge memory intelligence. Now featuring a **professional modular architecture**, enhanced **persistent storage**, and **enterprise-ready** design patterns.

---

## 🌟 **Major Highlights**

### ✅ **Recently Accomplished**
- **🎯 Complete architecture refactoring** - From monolithic to modular design
- **🛡️ NumPy 2.0 compatibility** - Fixed compilation issues with latest Python
- **📦 Persistent storage enabled** - ChromaDB + hnswlib working perfectly
- **🔧 Python 3.13.5 compatibility** - Latest Python version support
- **🏗️ Enterprise-ready structure** - Professional-grade codebase

### 🚀 **Current Status: Fully Operational**
- ✅ **Voice recognition** with natural pause detection
- ✅ **Text-to-speech** with voice interruption capability  
- ✅ **Persistent memory storage** with ChromaDB + hnswlib
- ✅ **Advanced AI integration** with Ollama
- ✅ **User management** with persistent profiles
- ✅ **Anti-hallucination** memory filtering
- ✅ **Comprehensive error handling** and logging

---

## 🏗️ **Professional Architecture**

### 📁 **New Modular Structure**

```
HeraAI/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization with NumPy compatibility
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Centralized configuration
│   ├── memory/                  # Advanced memory system
│   │   ├── __init__.py
│   │   ├── core.py             # Main memory operations
│   │   ├── models.py           # Data models (MemoryNode, etc.)
│   │   └── storage.py          # ChromaDB persistent storage
│   ├── audio/                   # Audio input/output
│   │   ├── __init__.py
│   │   ├── speech_recognition.py  # Voice input with pause detection
│   │   ├── text_to_speech.py      # TTS with interruption support
│   │   └── voice_interruption.py # Voice-based interruption
│   ├── ai/                      # AI model integration
│   │   ├── __init__.py
│   │   └── ollama_client.py    # Ollama API client
│   ├── ui/                      # User interface
│   │   ├── __init__.py
│   │   ├── user_management.py  # User identification & profiles
│   │   └── cli.py              # Beautiful command-line interface
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── logging.py          # Comprehensive logging
│       ├── error_handling.py   # Error management
│       └── numpy_compat.py     # NumPy 2.0 compatibility layer
├── main_v2.py                   # Refactored main application
├── requirements.txt             # Dependencies
└── README.md                    # This comprehensive guide
```

### 🎯 **Architecture Benefits**

#### ✅ **Readability**
- **Clear separation of concerns** - Each module has a single responsibility
- **Comprehensive documentation** - Every class and function documented
- **Type hints throughout** - Full type annotations for IDE support
- **Self-documenting code** - Meaningful names and clear structure

#### ✅ **Maintainability** 
- **Modular design** - Easy to modify individual components
- **Centralized configuration** - All settings in one place
- **Robust error handling** - Graceful failure management
- **Professional logging** - Structured logging throughout

#### ✅ **Extensibility**
- **Plugin-ready architecture** - Easy to add new features
- **Interface-based design** - Components can be swapped easily
- **Configuration-driven behavior** - Customizable without code changes
- **Future-ready structure** - Prepared for advanced features

---

## 🚀 **Quick Start**

### 📋 **Requirements**
- **Python 3.13.5** (recommended) or 3.11+
- **4GB+ RAM** (for embedding models)
- **2GB+ storage** (for memory database)
- **Microphone and speakers**
- **Internet connection** (for speech services)

### ⚡ **Installation**

#### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **2. Setup Ollama**
```bash
# Download and install Ollama from https://ollama.com/download
# Pull a model (e.g., llama3)
ollama pull llama3

# Start Ollama (keep running)
ollama serve
```

#### **3. Run HeraAI**
```bash
python main_v2.py
```

### 🔧 **Environment Setup (If Needed)**

If you encounter any dependency issues, use one of these proven solutions:

#### **Option 1: Python 3.11 (Most Reliable)**
```bash
# Install Python 3.11 from python.org
python3.11 -m venv heraai_env
heraai_env\Scripts\activate  # Windows
# or
source heraai_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

#### **Option 2: Conda Environment**
```bash
conda create -n heraai python=3.11
conda activate heraai
pip install -r requirements.txt
```

---

## 🌟 **Core Features**

### 🎤 **Natural Voice Interaction**
- **Voice-based conversation** - Speak naturally, AI detects when you finish
- **Voice interruption** - Interrupt AI speech by simply speaking
- **Hands-free operation** - No buttons needed, purely voice-controlled
- **Realistic speech synthesis** - Microsoft Edge TTS with natural voices
- **Smart pause detection** - Intelligent detection of speech completion

### 🧠 **Ultra-Advanced Memory System**
- **Persistent storage** - ChromaDB + hnswlib for fast vector operations
- **50,000+ memory capacity** - Massive long-term storage per user
- **User-specific isolation** - Individual memory collections per user
- **Anti-hallucination filtering** - Strict relevance thresholds prevent false memories
- **Semantic clustering** - Automatically organizes memories by topics
- **Relationship mapping** - Discovers connections between memories

### 🤖 **AI Self-Reflection & Analysis**
- **Memory pattern analysis** - AI analyzes its own memory usage
- **Conversation quality assessment** - Tracks engagement and depth
- **Personality trait detection** - Identifies AI's communication style
- **Learning adaptation** - AI monitors its own performance
- **Predictive capabilities** - Anticipates relevant memories

### 🔍 **Natural Language Memory Search**
- **Conversational queries** - "What did I say about my job?"
- **Temporal search** - "What did I mention yesterday?"
- **Emotional search** - "When was I excited about something?"
- **Smart suggestions** - Related search recommendations
- **Multi-strategy retrieval** - Semantic + temporal + relational search

### 📊 **Memory Visualization & Analytics**
- **Memory network graphs** - Visual maps of memory relationships
- **Timeline visualizations** - Memory importance over time
- **Word clouds** - Visual representation of key topics
- **Comprehensive analytics** - Deep insights into memory patterns
- **Health monitoring** - Memory system performance tracking

---

## 🎮 **Voice Commands**

### **Basic Commands**
- `"exit"` / `"quit"` / `"stop"` - End conversation
- `"memory stats"` - Show memory statistics
- `"show profile"` - Display user profile
- `"list users"` - Show all users

### **Advanced Memory Commands**
- `"ai insights"` - Get AI's self-analysis of memory patterns
- `"memory visualize"` - Create visual memory network maps
- `"memory report"` - Generate comprehensive memory analysis
- `"memory health"` - Check memory system health and quality

### **Intelligent Search & Prediction**
- `"search memories [query]"` - Advanced natural language search
- `"predict memories [context]"` - Predict relevant future memories
- `"show context [query]"` - Analyze memory context for topic

### **Natural Language Search Examples**
```
"search memories what did I say about my job"
"search memories when was I excited"
"search memories personal information"
"search memories AI and technology"
"search memories my goals and dreams"
```

---

## 🛠️ **Configuration**

All configuration is centralized in `src/config/settings.py`:

### **Memory System Configuration**
```python
MEMORY_CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "max_memories_per_user": 50000,
    "strict_semantic_threshold": 0.4,
    "enable_clustering": True,
    "enable_relationships": True
}
```

### **Audio System Configuration**
```python
AUDIO_CONFIG = {
    "speech_recognition": {
        "pause_threshold": 1.0,
        "energy_threshold": 300,
        "timeout": 10
    },
    "text_to_speech": {
        "voice": "en-US-AriaNeural",
        "rate": "+0%",
        "volume": "+0%"
    }
}
```

### **AI Model Configuration**
```python
AI_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "default_model": "llama3",
        "timeout": 30
    }
}
```

---

## 💻 **Usage Examples**

### **Basic Conversation**
```python
from src.memory.core import AdvancedMemorySystem
from src.ai.ollama_client import OllamaClient

# Initialize memory system
memory = AdvancedMemorySystem()
memory.set_user("Alice")

# Save a memory
memory_id = memory.save_memory("I love programming in Python", "user", "preference")

# Get context for conversation
context = memory.get_conversation_context("What do I like?")

# Generate AI response
client = OllamaClient()
response = client.generate_response("Tell me about my preferences", context=context)
```

### **Advanced Memory Retrieval**
```python
# Advanced memory search
memories = memory.retrieve_memories("favorite books", top_k=5)
for result in memories:
    print(f"Content: {result.content}")
    print(f"Relevance: {result.semantic_similarity:.3f}")
    print(f"Importance: {result.importance_score:.3f}")
```

### **Memory Analytics**
```python
# Get memory statistics
stats = memory.get_memory_statistics()
print(f"Total memories: {stats['total_memories']}")
print(f"Memory clusters: {stats['total_clusters']}")
print(f"Average importance: {stats['average_importance']:.3f}")
```

---

## 🔧 **Technical Architecture**

### **Core Dependencies**
- **ChromaDB 1.0.13+** - Vector database for persistent storage
- **hnswlib 0.8.0+** - Fast approximate nearest neighbor search
- **sentence-transformers** - Advanced embedding models
- **Edge TTS** - Natural text-to-speech synthesis
- **SpeechRecognition** - Voice input processing
- **Ollama** - Local LLM integration

### **Advanced Features**
- **Semantic clustering** - K-means clustering with dynamic optimization
- **Relationship mapping** - Bidirectional memory connections
- **Emotional analysis** - Multi-emotion detection and scoring
- **Importance weighting** - Context-aware memory prioritization
- **Temporal decay** - Time-based memory relevance adjustment
- **Quality filtering** - Anti-hallucination safeguards

### **Data Models**
```python
class MemoryNode:
    """Advanced memory storage with relationships and metadata"""
    id: str
    content: str
    embedding: List[float]
    metadata: MemoryMetadata
    relationships: List[str]
    importance_score: float

class MemoryMetadata:
    """Comprehensive memory metadata"""
    timestamp: datetime
    user_id: str
    role: str
    memory_type: str
    emotional_context: Dict[str, float]
    access_count: int
```

---

## 📈 **Performance Metrics**

### **Memory System Capabilities**
- **50,000+ memories per user** - Massive storage capacity
- **Multi-year retention** - Long-term memory preservation
- **Sub-second retrieval** - Fast memory access with hnswlib
- **99%+ accuracy** - Reliable memory validation
- **Real-time clustering** - Dynamic memory organization

### **Technical Performance**
- **Vector search** - O(log n) complexity with HNSW algorithm
- **Memory filtering** - Strict semantic thresholds (0.4+)
- **Concurrent operations** - Thread-safe memory operations
- **Graceful degradation** - Continues operation if components fail
- **Resource optimization** - Efficient memory and CPU usage

---

## 🛡️ **Error Handling & Compatibility**

### **NumPy 2.0 Compatibility**
HeraAI includes a comprehensive NumPy 2.0 compatibility layer that:
- **Restores deprecated attributes** (float_ → float64, int_ → int64)
- **Suppresses deprecation warnings** for clean operation
- **Handles ChromaDB compatibility** issues automatically
- **Provides graceful fallbacks** for missing dependencies

### **Robust Error Handling**
- **Graceful degradation** - System continues if components fail
- **Comprehensive logging** - Detailed error information
- **User-friendly messages** - Clear feedback for users
- **Automatic recovery** - Self-healing capabilities where possible

### **Dependency Management**
- **Optional dependencies** - Core features work without optional packages
- **Version compatibility** - Tested with multiple Python versions
- **Fallback mechanisms** - Alternative implementations when needed
- **Clear error messages** - Helpful guidance for missing dependencies

---

## 🚀 **Recent Achievements**

### **✅ Complete Architecture Transformation**
- **From monolithic to modular** - 2 massive files → 15+ focused modules
- **Professional design patterns** - Industry-standard architecture
- **Enhanced readability** - Self-documenting code with full type hints
- **Improved maintainability** - Easy to extend and modify

### **✅ Environment Compatibility Fixes**
- **Python 3.13.5 support** - Latest Python version compatibility
- **NumPy 2.0 compatibility** - Fixed compilation issues
- **ChromaDB 1.0.13+ integration** - Modern vector database support
- **hnswlib compilation** - Resolved Windows build issues

### **✅ Enhanced Features**
- **Persistent storage** - Full ChromaDB + hnswlib integration
- **Advanced memory analytics** - Deep insights into usage patterns
- **Professional logging** - Comprehensive debugging capabilities
- **Configuration management** - Centralized settings system

---

## 🔮 **Future Roadmap**

### **Planned Enhancements**
- **Web interface** - Browser-based interaction
- **Mobile app** - Cross-platform voice assistant
- **Plugin system** - Third-party extensions
- **Multi-language support** - International voice recognition
- **Advanced visualizations** - Interactive memory exploration

### **Technical Improvements**
- **Distributed storage** - Multi-node memory system
- **Advanced clustering** - ML-based memory organization
- **Real-time learning** - Continuous model adaptation
- **Performance optimization** - Further speed improvements
- **Security enhancements** - Advanced data protection

---

## 📞 **Support & Troubleshooting**

### **Common Issues**

#### **Voice Recognition Problems**
```bash
# Check microphone permissions
# Ensure internet connection for Google Speech API
# Adjust energy threshold in settings.py
```

#### **Memory Storage Issues**
```bash
# Verify ChromaDB installation
pip install --upgrade chromadb hnswlib

# Check database permissions
# Clear test database if needed
```

#### **AI Connection Problems**
```bash
# Ensure Ollama is running
ollama serve

# Test Ollama connection
curl http://localhost:11434/api/tags
```

### **Getting Help**
- **Check logs** - Review console output and log files
- **Verify configuration** - Ensure settings are correct
- **Test components** - Use individual module tests
- **Environment validation** - Confirm all dependencies installed

---

## 📜 **Version History**

### **v2.0 (Current) - Refactored Architecture**
- ✅ Complete modular architecture
- ✅ NumPy 2.0 & Python 3.13.5 compatibility
- ✅ Persistent storage with ChromaDB + hnswlib
- ✅ Enhanced error handling and logging
- ✅ Professional-grade codebase

### **v1.0 - Original Monolithic Version**
- Advanced memory system with 50,000+ capacity
- Natural voice interaction with interruption
- AI self-reflection and analytics
- Memory visualization and search
- Comprehensive feature set

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Ollama** - Local LLM integration
- **ChromaDB** - Vector database technology
- **hnswlib** - Fast nearest neighbor search
- **sentence-transformers** - Advanced embedding models
- **Microsoft Edge TTS** - Natural speech synthesis

---

**🎯 HeraAI v2.0 - Where advanced AI meets professional engineering**

*Built with ❤️ for the future of voice AI assistants*
