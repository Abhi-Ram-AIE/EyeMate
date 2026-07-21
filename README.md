# EyeMate 👁️

EyeMate is an AI-powered assistive application that helps visually impaired users by providing:

* 📝 Text reading (OCR)
* 🗣️ Text-to-Speech (TTS)
* 🎤 Voice-triggered interaction
* 🚶 Object detection using YOLOv8 and MobileNet SSD

## Features

* OCR-based text extraction
* Voice output using Text-to-Speech
* Real-time object detection
* Voice-triggered commands
* GUI-based interface

---

## Prerequisites

* Python 3.10 or later
* Git
* Webcam
* Internet connection (if using cloud OCR or online TTS)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhi-Ram-AIE/EyeMate.git
cd EyeMate
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download Required Models

The following model files are not included in this repository because they are large:

* `yolov8n.pt`
* `yolov8m.pt`
* `yolov8n-oiv7.pt`
* `yolov8m-oiv7.pt`

Download them from the official Ultralytics model releases or generate them as required by your project, then place them in the project root (or the location expected by the code).

---

## Google Cloud OCR Credentials

This repository does **not** include Google Cloud credentials.

To use OCR:

1. Create a Google Cloud project.
2. Enable the Vision API.
3. Create a Service Account.
4. Download the JSON credentials.
5. Save the file as:

```
eyemate-ocr.json
```

Place it in the project root.

Alternatively, set the environment variable:

```text
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
```

---

## Run the Application

Run the main application:

```bash
python eyemate_main.py
```

or

```bash
python eyemate_gui.py
```

depending on the interface you want to use.

---

## Project Structure

```text
EyeMate/
│
├── models/
├── object-detection-files/
├── eyemate_main.py
├── eyemate_gui.py
├── text_reader.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* OpenCV
* YOLOv8
* MobileNet SSD
* Google Cloud Vision API
* gTTS
* SpeechRecognition

---

## License

This project is licensed under the MIT License.
