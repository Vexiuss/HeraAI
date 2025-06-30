import os
import requests
import speech_recognition as sr
import pyttsx3
import asyncio
import edge_tts

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def ask_gpt(prompt):
    # Use Ollama's local API
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",  # Change to your preferred model if needed
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Sorry, I couldn't generate a response.")
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "Sorry, I couldn't connect to the local AI model."

def speak(text):
    print(f"AI: {text}")
    # Use Edge TTS for realistic speech
    voice = "en-US-JennyNeural"  # You can change to any supported voice
    async def _speak():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save("output.mp3")
    asyncio.run(_speak())
    # Play the audio file
    import platform
    import subprocess
    if platform.system() == "Windows":
        subprocess.run(["powershell", "-c", "(New-Object Media.SoundPlayer 'output.mp3').PlaySync(); Remove-Item 'output.mp3'"])
    else:
        subprocess.run(["mpg123", "output.mp3"])
        os.remove("output.mp3")

def main():
    print("--- HeraAI Voice Chat ---")
    conversation = []  # Store conversation history
    while True:
        print("Listening...")
        user_input = listen()
        if user_input:
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Goodbye!")
                break
            conversation.append({"role": "user", "content": user_input})
            print("Thinking...")
            # Build context for the model
            context = "\n".join([f"User: {msg['content']}" if msg['role']=='user' else f"AI: {msg['content']}" for msg in conversation])
            ai_response = ask_gpt(context)
            conversation.append({"role": "ai", "content": ai_response})
            speak(ai_response)

if __name__ == "__main__":
    main()
