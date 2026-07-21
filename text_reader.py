import cv2
import easyocr
import pygame
import time
import speech_recognition as sr
from gtts import gTTS
import numpy as np
import re
import os
from spellchecker import SpellChecker

# Initialize recognizer and microphone
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Initialize EasyOCR reader (English only)
reader = easyocr.Reader(["en"], gpu=False)

# Initialize pygame mixer for audio
pygame.mixer.init()

# Initialize SpellChecker
spell = SpellChecker()

print("[INFO] EasyOCR Voice-Controlled Text Reader with Auto-Correction is running. Say 'scan' to read the visible text.")

# Confidence threshold for OCR
CONFIDENCE_THRESHOLD = 0.6

# Text cleanup function
def clean_text(text):
    """Remove junk characters and normalize spacing."""
    text = re.sub(r"[\n\r]+", " ", text)           # remove newlines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)     # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()       # normalize spaces
    return text

# Spell correction function
def correct_spelling(text):
    """Correct spelling word by word using pyspellchecker."""
    corrected_words = []
    for word in text.split():
        if word.isalpha():  # correct only alphabetic words
            corrected_words.append(spell.correction(word.lower()))
        else:
            corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text.capitalize()

# Speak function
def speak(text):
    try:
        filename = "read_text.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[ERROR] Speaking failed: {e}")

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

    valid_lines = []
    for (bbox, text, confidence) in results:
        if confidence >= CONFIDENCE_THRESHOLD:
            cleaned = clean_text(text)
            if cleaned:
                valid_lines.append(cleaned)

    if valid_lines:
        combined_text = " ".join(valid_lines)
        corrected_text = correct_spelling(combined_text)
        print("[INFO] Extracted Text:", corrected_text)
        speak(corrected_text)
    else:
        print("[INFO] No high-confidence text found.")
        speak("Sorry, no readable text found.")

# Listen for command loop
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
