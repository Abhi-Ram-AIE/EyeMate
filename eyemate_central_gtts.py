import cv2
import time
import os
import uuid
from collections import Counter
from ultralytics import YOLO
from gtts import gTTS
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load YOLOv8 Open Images model
model = YOLO("yolov8n-oiv7.pt")

# Start video capture
cap = cv2.VideoCapture(0)
print("[INFO] EyeMate (center view, gTTS with plural narration) is running.")

last_spoken_labels = {}
SPEAK_DELAY = 2  # seconds between announcements

def speak(text):
    try:
        filename = f"speak_{uuid.uuid4().hex}.mp3"
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx1, cy1, cx2, cy2 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    center_crop = frame[cy1:cy2, cx1:cx2]
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    results = model(center_crop)[0]
    labels_detected = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        labels_detected.append(label)

    label_counts = Counter(labels_detected)

    for label, count in label_counts.items():
        last_time = last_spoken_labels.get(label, 0)
        if time.time() - last_time > SPEAK_DELAY:
            if count == 1:
                message = {
                    "person": "A person is ahead.",
                    "bottle": "There is a bottle.",
                    "chair": "A chair is nearby.",
                    "dog": "A dog is in front.",
                    "cat": "A cat is visible.",
                    "car": "A car is approaching.",
                    "tv": "You are looking at a TV.",
                    "laptop": "A laptop is in front of you.",
                    "cell phone": "A phone is ahead."
                }.get(label, f"A {label} is in front of you.")
            else:
                plural_label = label + "s" if not label.endswith("s") else label
                message = f"{count} {plural_label} are in front of you."

            speak(message)
            last_spoken_labels[label] = time.time()

    cv2.imshow("EyeMate - Central Region Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
