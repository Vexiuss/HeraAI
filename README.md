# HeraAI

# HeraAI Voice Chat

This project is an AI voice chat assistant using open-source LLMs (via Ollama) and realistic speech synthesis (Edge TTS).

## Features
- Real-time voice chat with AI
- Local LLM (Llama 3 or compatible) via Ollama
- Realistic text-to-speech using Microsoft Edge TTS
- Conversation history for context

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies
- Ollama (https://ollama.com/) with a model (e.g., llama3) pulled and running

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install and run Ollama:
   - Download from https://ollama.com/download
   - Pull a model (e.g., `ollama pull llama3`)
   - Start Ollama: `ollama run llama3`

## Usage
- To start the voice chat AI:
  ```
  python main.py
  ```
- To test the TTS voice only:
  ```
  python test_tts.py
  ```

## Customization
- Change the voice in `main.py` or `test_tts.py` by editing the `voice` variable (see Edge TTS docs for available voices).

---

