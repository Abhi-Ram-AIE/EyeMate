from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Real-time detection and speech started. Press 'q' to quit.")

last_spoken = 0
DELAY_BETWEEN_SPEECH = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame)[0]
    spoken = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Speak every object once per frame with a short delay between frames
        if current_time - last_spoken > DELAY_BETWEEN_SPEECH:
            description = {
                "person": "a person",
                "bottle": "a bottle",
                "chair": "a chair",
                "dog": "a dog",
                "cat": "a cat",
                "car": "a car",
                "bus": "a bus",
                "tv": "a TV",
                "laptop": "a laptop",
                "cell phone": "a phone"
            }.get(label, f"a {label}")

            engine.say(f"I see {description}")
            engine.runAndWait()
            last_spoken = current_time
            spoken = True

    cv2.imshow("EyeMate - YOLOv8 Per Object Speech", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
