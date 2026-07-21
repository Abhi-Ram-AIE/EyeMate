import cv2
import easyocr
import pygame
import time
import speech_recognition as sr
from gtts import gTTS
import numpy as np
import re

# Initialize
recognizer = sr.Recognizer()
mic = sr.Microphone()
reader = easyocr.Reader(["en"], gpu=False)
pygame.mixer.init()

print("[INFO] EasyOCR Voice-Controlled Text Reader is running. Say 'scan' to read the visible text.")

# Text cleanup function
def clean_text(text):
    # Remove junk characters and fix spacing
    text = re.sub(r"[\n\r]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text.capitalize()

# Speak function
def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        filename = "read_text.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print("[ERROR] Speaking failed:", e)

# Capture and read text
def capture_and_read_text():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    print("[INFO] Capturing frame...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Failed to capture frame.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    extracted = " ".join([text[1] for text in results])

    cleaned = clean_text(extracted)
    print("[INFO] Extracted Text:", cleaned)
    if cleaned:
        speak(cleaned)
    else:
        speak("Sorry, no readable text found.")

# Listen for command
while True:
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("\n[LISTENING] Say 'scan' to capture and read text (or say 'exit' to quit)...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"[YOU SAID] {command}")

            if "scan" in command:
                capture_and_read_text()
            elif "exit" in command:
                print("[INFO] Exiting program.")
                break

        except sr.WaitTimeoutError:
            print("[WARNING] Listening timed out. No command detected.")
        except sr.UnknownValueError:
            print("[WARNING] Could not understand the audio.")
        except sr.RequestError as e:
            print(f"[ERROR] Could not request results from Google Speech Recognition service; {e}")