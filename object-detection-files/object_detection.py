import cv2
import pyttsx3
import numpy as np

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load the pretrained model
net = cv2.dnn.readNetFromCaffe(
    "models/mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    "models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = set()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_objects.add(label)

    # Create and speak a combined voice message
    if detected_objects:
        sentence_parts = []
        for label in detected_objects:
            if label == "person":
                sentence_parts.append("a person")
            elif label == "bottle":
                sentence_parts.append("a bottle")
            elif label == "chair":
                sentence_parts.append("a chair")
            elif label == "dog":
                sentence_parts.append("a dog")
            elif label == "cat":
                sentence_parts.append("a cat")
            elif label == "bus":
                sentence_parts.append("a bus")
            elif label == "car":
                sentence_parts.append("a car")
            elif label == "train":
                sentence_parts.append("a train")
            elif label == "tvmonitor":
                sentence_parts.append("a TV or monitor")
            elif label == "sofa":
                sentence_parts.append("a sofa")
            elif label == "pottedplant":
                sentence_parts.append("a potted plant")
            else:
                sentence_parts.append(f"a {label}")

        description = " and ".join(sentence_parts)
        engine.say(f"I see {description}")
        engine.runAndWait()

    # Show webcam frame
    cv2.imshow("EyeMate - Object Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
