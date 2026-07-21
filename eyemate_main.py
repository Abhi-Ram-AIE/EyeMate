import cv2
import threading
import time
import os
import uuid
import numpy as np
import easyocr
import pygame
from gtts import gTTS
import speech_recognition as sr
from textblob import TextBlob
from ultralytics import YOLO

# Init modules
pygame.mixer.init()
reader = easyocr.Reader(["en"], gpu=False)
model = YOLO("yolov8m.pt")
recognizer = sr.Recognizer()
mic = sr.Microphone()

running = True  # main control flag

# Speak function
def speak(text):
    try:
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[ERROR] Failed to speak: {e}")

# Text cleanup + autocorrect
def spell_correct(text):
    words = text.split()
    corrected = []
    for word in words:
        blob = TextBlob(word)
        corrected_word = str(blob.correct())
        corrected.append(corrected_word)
    return " ".join(corrected)

# Capture & OCR
def capture_and_read_text():
    print("[INFO] Scanning text now...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Failed to capture image.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    extracted = " ".join([text[1] for text in results])

    if extracted:
        corrected = spell_correct(extracted)
        print("[TEXT DETECTED]:", corrected)
        speak(corrected)
    else:
        speak("No clear text detected.")

# Object detection loop
def object_detection_loop():
    cap = cv2.VideoCapture(0)
    print("[INFO] EyeMate Object Detection started.")
    last_spoken_time = 0.0
    narration_cooldown_sec = 3.0

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        cx1, cy1, cx2, cy2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
        center_crop = frame[cy1:cy2, cx1:cx2]

        results = model(center_crop)[0]

        largest_box = None
        largest_area = 0
        label_to_speak = ""

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                label_to_speak = label
                largest_box = (x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1)

        if label_to_speak:
            now = time.time()
            if now - last_spoken_time >= narration_cooldown_sec:
                narration = {
                    "person": "A person is ahead.",
                    "bottle": "There is a bottle.",
                    "book": "A book is ahead.",
                    "stop sign": "There is a stop sign.",
                    "laptop": "A laptop is visible.",
                    "cell phone": "A phone is in front.",
                }.get(label_to_speak, f"A {label_to_speak} is ahead.")
                speak(narration)
                last_spoken_time = now

        time.sleep(0.5)

    cap.release()
    print("[INFO] Object detection stopped.")

# Voice command listener
def listen_for_commands():
    global running
    print("[INFO] Say 'scan' to read text or 'exit' to quit.")

    while running:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"[YOU SAID] {command}")

                if "scan" in command:
                    threading.Thread(target=capture_and_read_text).start()

                elif "exit" in command:
                    print("[INFO] Exit command received.")
                    running = False
                    break

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("[WARN] Didn't catch that.")
            except sr.RequestError as e:
                print(f"[ERROR] Voice recognition error: {e}")

    print("[INFO] Voice command listener stopped.")

# Main
if __name__ == "__main__":
    t1 = threading.Thread(target=object_detection_loop)
    t2 = threading.Thread(target=listen_for_commands)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("[INFO] EyeMate application exited.")
