# HeraAI

# HeraAI Voice Chat Prototype

This is a simple prototype for a voice chat AI using Python. It uses:
- SpeechRecognition + PyAudio for speech-to-text
- OpenAI GPT-4 for conversational AI
- pyttsx3 for text-to-speech (offline)

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key as an environment variable:
   - On Windows (PowerShell):
     ```
     $env:OPENAI_API_KEY="your-api-key-here"
     ```

## Usage
Run the main script:
```
python main.py
```

Speak into your microphone and the AI will respond with voice.

