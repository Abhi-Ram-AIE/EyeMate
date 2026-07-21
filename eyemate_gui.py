import cv2
import threading
import tkinter as tk
from tkinter import Label, Button, StringVar
from ultralytics import YOLO
import easyocr
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import time
from textblob import TextBlob
from PIL import Image, ImageTk
from collections import Counter

# Initialize all components
pygame.mixer.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()
model = YOLO("yolov8m.pt")
reader = easyocr.Reader(["en"], gpu=False)

# Global states
running = False
voice_running = False
scanning = False
current_frame = None
cap = None
last_narration_time = 0.0
NARRATION_COOLDOWN_SEC = 3.0

# GUI setup
window = tk.Tk()
window.title("EyeMate: AI Visual Assistant")
window.geometry("900x700")
window.configure(bg="#101820")

status_text = StringVar(value="Status: Idle")
narration_text = StringVar(value="Narration: ---")
listening_text = StringVar(value="🎤 Status: Not Listening")

Label(window, text="EyeMate: AI Visual Assistant", font=("Arial", 20, "bold"), bg="#101820", fg="#FEE715").pack(pady=10)
video_label = Label(window)
video_label.pack()
Label(window, textvariable=narration_text, font=("Arial", 14), bg="#101820", fg="#FFFFFF").pack(pady=5)
Label(window, textvariable=status_text, font=("Arial", 12), bg="#101820", fg="#A5A5A5").pack()
Label(window, textvariable=listening_text, font=("Arial", 12, "bold"), bg="#101820", fg="#FEE715").pack(pady=5)

# ---------- Core Functions ----------

def speak(text):
    """Convert text to speech using gTTS"""
    try:
        if not text.strip():
            return
        filename = "tts_output.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print("[ERROR] Speaking:", e)

def spell_correct(text):
    """Auto correct OCR output"""
    words = text.split()
    corrected = []
    for w in words:
        try:
            corrected.append(str(TextBlob(w).correct()))
        except:
            corrected.append(w)
    return " ".join(corrected)

def capture_and_read_text(frame):
    """Capture text from frame and read aloud"""
    global scanning
    scanning = True
    status_text.set("Status: Scanning text...")
    listening_text.set("🎤 Status: Not Listening")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    extracted = " ".join([t[1] for t in results if t[2] > 0.5])
    corrected = spell_correct(extracted)
    final_text = corrected if corrected.strip() else "No readable text found."
    narration_text.set(f"Narration: {final_text}")
    speak(f"Text detected: {final_text}")

    scanning = False
    status_text.set("Status: Object detection running")
    listening_text.set("🎤 Status: Listening...")

def voice_listener():
    """Listen for voice commands"""
    global running, voice_running
    while running and voice_running:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            listening_text.set("🎤 Status: Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                listening_text.set("🎤 Status: Processing...")
                command = recognizer.recognize_google(audio).lower().strip()
                print(f"[YOU SAID] {command}")

                if "scan" in command and current_frame is not None and not scanning:
                    threading.Thread(target=capture_and_read_text, args=(current_frame.copy(),), daemon=True).start()
                elif "exit" in command:
                    print("[INFO] Exit command received.")
                    stop_detection()
                    return_to_idle()
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print("[ERROR] Voice:", e)

def start_detection():
    """Start YOLOv8 real-time object detection with plural + direction support"""
    global running, cap, current_frame, last_narration_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.set("Status: Could not open camera (index 0).")
        cap = None
        running = False
        return
    status_text.set("Status: Object detection running")

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        current_frame = frame.copy()
        results = model(frame)
        h, w, _ = frame.shape

        # Get all detections
        labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
        counts = Counter(labels)

        # For narration (once per cooldown, not every frame — narrated_labels was reset each frame before)
        narrated_labels = []
        narration_phrases = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)

            # Direction
            if cx < w / 3:
                direction = "on your left"
            elif cx > 2 * w / 3:
                direction = "on your right"
            else:
                direction = "in front"

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label not in narrated_labels:
                count = counts[label]
                if count == 1:
                    text = f"A {label} is {direction}."
                else:
                    text = f"{count} {label}s are {direction}."
                narration_phrases.append(text)
                narrated_labels.append(label)

        now = time.time()
        if narration_phrases and (now - last_narration_time) >= NARRATION_COOLDOWN_SEC:
            combined = " ".join(narration_phrases)
            last_narration_time = now
            narration_text.set(f"Narration: {combined}")
            threading.Thread(target=speak, args=(combined,), daemon=True).start()

        # Display feed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        window.update_idletasks()
        window.update()

    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()

def start_app():
    """Start both detection and voice listener"""
    global running, voice_running
    running = True
    voice_running = True
    status_text.set("Status: Listening for commands...")
    listening_text.set("🎤 Status: Listening...")
    threading.Thread(target=voice_listener, daemon=True).start()
    threading.Thread(target=start_detection, daemon=True).start()

def stop_detection():
    """Stop detection (camera is released by the detection thread when the loop exits)."""
    global running
    running = False
    status_text.set("Status: Object detection stopped.")
    listening_text.set("🎤 Status: Not Listening")

def return_to_idle():
    """Reset app"""
    global running, voice_running
    running = False
    voice_running = False
    status_text.set("Status: Idle")
    listening_text.set("🎤 Status: Not Listening")
    narration_text.set("Narration: ---")
    speak("Returning to idle state.")

def exit_app():
    """Exit the app"""
    global running, voice_running
    running = False
    voice_running = False
    listening_text.set("🎤 Status: Not Listening")
    speak("EyeMate shutting down.")
    window.destroy()

# Buttons
Button(window, text="Start EyeMate", command=start_app, bg="#FEE715", fg="#101820", font=("Arial", 12, "bold")).pack(pady=10)
Button(window, text="Exit Application", command=exit_app, bg="#B22222", fg="white", font=("Arial", 12, "bold")).pack(pady=5)

window.mainloop()
