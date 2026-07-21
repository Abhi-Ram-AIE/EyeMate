
import cv2
import time
import os
import uuid
import threading
import numpy as np
import pygame
import easyocr
import speech_recognition as sr
from gtts import gTTS
from textblob import TextBlob
from ultralytics import YOLO

# ----------------------- Initialization ---------------------------
pygame.mixer.init()
reader = easyocr.Reader(["en"], gpu=False)
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Load YOLO model
model = YOLO("yolov8m.pt")

# Globals
exit_event = threading.Event()
scan_trigger = threading.Event()

# ----------------------- Utility Functions ---------------------------

def speak(text):
    try:
        filename = "tts_audio.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[ERROR] Failed to speak: {e}")

def spell_correct(text):
    words = text.split()
    corrected = [str(TextBlob(w).correct()) for w in words if w]
    return " ".join(corrected)

# ----------------------- Text Reader ---------------------------
def text_reader_once():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not capture image.")
        speak("Failed to capture image.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    extracted = " ".join([text[1] for text in results])
    cleaned = extracted.strip()

    if cleaned:
        corrected = spell_correct(cleaned)
        print(f"[TEXT DETECTED]: {corrected}")
        speak(corrected)
    else:
        print("[TEXT DETECTED]: No text found.")
        speak("No readable text found.")

# ----------------------- Object Detection Loop ---------------------------
def object_detection_loop():
    cap = cv2.VideoCapture(0)
    print("[INFO] EyeMate Object Detection started.")
    last_spoken = ""
    last_spoken_time = 0
    delay = 3

    while not exit_event.is_set():
        if scan_trigger.is_set():
            cap.release()
            print("[INFO] Scanning text now...")
            text_reader_once()
            scan_trigger.clear()
            cap = cv2.VideoCapture(0)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        cx1, cy1, cx2, cy2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
        center_crop = frame[cy1:cy2, cx1:cx2]
        results = model(center_crop)[0]

        counts = {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            counts[label] = counts.get(label, 0) + 1

        narration = None
        for label, count in counts.items():
            if count == 1:
                narration = f"There is a {label}."
            else:
                narration = f"There are {count} {label}s."
            break  # Speak only one object

        if narration and (time.time() - last_spoken_time > delay or narration != last_spoken):
            speak(narration)
            last_spoken = narration
            last_spoken_time = time.time()

    cap.release()
    print("[INFO] Object detection stopped.")

# ----------------------- Voice Listener Thread ---------------------------
def voice_listener_loop():
    print("[INFO] Say 'scan' to read text or 'exit' to quit.")
    while not exit_event.is_set():
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"[YOU SAID] {command}")

                if "scan" in command:
                    scan_trigger.set()
                elif "exit" in command:
                    print("[INFO] Exit command received.")
                    exit_event.set()
                    break
            except sr.UnknownValueError:
                print("[WARN] Didn't catch that.")
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"[ERROR] Voice recognition failed: {e}")

    print("[INFO] Voice command listener stopped.")

# ----------------------- Main ---------------------------
if __name__ == "__main__":
    detect_thread = threading.Thread(target=object_detection_loop)
    listen_thread = threading.Thread(target=voice_listener_loop)

    detect_thread.start()
    listen_thread.start()

    detect_thread.join()
    listen_thread.join()

    print("[INFO] EyeMate application exited.")
