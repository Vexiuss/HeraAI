import asyncio
import edge_tts
import pygame
import time
import os

# Test text and voice
text = "Hello! This is a test of the AI's voice using Edge TTS. How natural do I sound?"
voice = "en-US-JennyNeural"  # You can change to any supported voice

async def test_tts():
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("test_output.mp3")

if __name__ == "__main__":
    asyncio.run(test_tts())
    pygame.mixer.init()
    pygame.mixer.music.load("test_output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    os.remove("test_output.mp3")
