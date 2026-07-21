import cv2, threading, tkinter as tk
from tkinter import Label, Button, StringVar
from ultralytics import YOLO
import easyocr, speech_recognition as sr
from gtts import gTTS
import pygame, os, time, uuid
from textblob import TextBlob
from PIL import Image, ImageTk
from collections import Counter
from datetime import datetime

# ---------------- Initialization ----------------
pygame.mixer.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()
model = YOLO("yolov8m.pt", verbose=False)
reader = easyocr.Reader(["en"], gpu=False)

running = False
voice_running = False
scanning = False
current_frame = None
cap = None
last_spoken = {}
watchdog_alive = True

# ---------------- GUI ----------------
window = tk.Tk()
window.title("EyeMate: AI Visual Assistant")
window.geometry("900x700")
window.configure(bg="#101820")

status_text = StringVar(value="Status: Idle")
narration_text = StringVar(value="Narration: ---")
listening_text = StringVar(value="🎤 Status: Not Listening")

Label(window, text="EyeMate: AI Visual Assistant",
      font=("Arial", 20, "bold"), bg="#101820", fg="#FEE715").pack(pady=10)
video_label = Label(window, bg="#101820")
video_label.pack()
Label(window, textvariable=narration_text, font=("Arial", 14),
      bg="#101820", fg="#FFFFFF").pack(pady=5)
Label(window, textvariable=status_text, font=("Arial", 12),
      bg="#101820", fg="#A5A5A5").pack()
Label(window, textvariable=listening_text, font=("Arial", 12, "bold"),
      bg="#101820", fg="#FEE715").pack(pady=5)

# ---------------- Safe Tk Updates ----------------
def safe_set(v, val): window.after(0, lambda: v.set(val))
def safe_img(imgtk):
    def _(): video_label.imgtk = imgtk; video_label.configure(image=imgtk)
    window.after(0, _)

# ---------------- Audio ----------------
def speak(text):
    try:
        if not text.strip(): return
        fn = f"tts_{uuid.uuid4().hex}.mp3"
        gTTS(text=text.strip(), lang="en").save(fn)
        pygame.mixer.music.load(fn)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()
        os.remove(fn)
    except Exception as e: print("[AUDIO]", e)

def speak_async(text): threading.Thread(target=speak, args=(text,), daemon=True).start()

# ---------------- OCR ----------------
def spell_correct(t):
    out=[]
    for w in t.split():
        try: out.append(str(TextBlob(w).correct()))
        except: out.append(w)
    return " ".join(out)

def capture_and_read_text(frame):
    global scanning
    if scanning: return
    scanning=True
    safe_set(status_text,"Status: Scanning text...")
    safe_set(listening_text,"🎤 Status: Not Listening")
    try:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        res=reader.readtext(gray)
        texts=[t[1] for t in res if isinstance(t[1],str) and t[2]>=0.5]
        if not texts:
            safe_set(narration_text,"Narration: No readable text found.")
            speak_async("No readable text found.")
        else:
            corrected=spell_correct(" ".join(texts).strip())
            safe_set(narration_text,f"Narration: Readable text found: {corrected}")
            speak_async(f"Readable text found: {corrected}")
    except Exception as e:
        print("[OCR]", e)
        speak_async("I encountered an error while scanning.")
    finally:
        scanning=False
        if running:
            safe_set(status_text,"Status: Object detection running")
            safe_set(listening_text,"🎤 Status: Listening...")
        else:
            safe_set(status_text,"Status: Idle")
            safe_set(listening_text,"🎤 Status: Not Listening")

# ---------------- Voice ----------------
def voice_listener():
    global running, voice_running
    with mic as src:
        try: recognizer.adjust_for_ambient_noise(src, duration=1)
        except: pass
    while running and voice_running:
        try:
            with mic as src:
                safe_set(listening_text,"🎤 Status: Listening...")
                audio=recognizer.listen(src,timeout=5,phrase_time_limit=4)
            safe_set(listening_text,"🎤 Status: Processing...")
            try: cmd=recognizer.recognize_google(audio).lower().strip()
            except sr.UnknownValueError: continue
            except sr.RequestError: continue
            print("[VOICE]",cmd)
            if "scan" in cmd and current_frame is not None and not scanning:
                threading.Thread(target=capture_and_read_text,
                                 args=(current_frame.copy(),),daemon=True).start()
            elif "exit" in cmd:
                speak_async("EyeMate application closing.")
                stop_detection(); return_to_idle(); break
        except sr.WaitTimeoutError: continue
        except Exception as e: print("[VOICE]", e); time.sleep(0.3)
    # auto-restart listener if still running
    if running and not voice_running:
        threading.Thread(target=voice_listener,daemon=True).start()

# ---------------- Detection ----------------
def detect_loop():
    global cap, current_frame, last_spoken
    last_spoken.clear()
    try:
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): raise Exception("Camera not accessible")
        safe_set(status_text,"Status: Object detection running")
        while running:
            try:
                ret,frame=cap.read()
                if not ret: continue
                current_frame=frame.copy()
                h,w,_=frame.shape
                # Large center region (80%)
                l,r=int(w*0.1),int(w*0.9)
                t,b=int(h*0.1),int(h*0.9)
                results=model(frame)
                labels=[]; boxes=[]
                for bx in results[0].boxes:
                    x1,y1,x2,y2=map(int,bx.xyxy[0])
                    cx,cy=(x1+x2)//2,(y1+y2)//2
                    if l<=cx<=r and t<=cy<=b:
                        label=model.names[int(bx.cls[0])]
                        conf=float(bx.conf[0])
                        if conf>=0.35:
                            labels.append(label)
                            boxes.append((label,conf,x1,y1,x2,y2,cx))
                counts=Counter(labels)
                now=datetime.now()
                for label,conf,x1,y1,x2,y2,cx in boxes:
                    last=last_spoken.get(label)
                    if last and (now-last).total_seconds()<5: continue
                    last_spoken[label]=now
                    direction="in front"
                    if cx<w/3: direction="on your left"
                    elif cx>2*w/3: direction="on your right"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{label} ({conf:.2f})",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                    c=counts[label]
                    text=f"A {label} is {direction}." if c==1 else f"{c} {label}s are {direction}."
                    safe_set(narration_text,f"Narration: {text}")
                    speak_async(text)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                imgtk=ImageTk.PhotoImage(Image.fromarray(rgb))
                safe_img(imgtk)
            except Exception as e:
                print("[LOOP]",e); time.sleep(0.2)
        cap.release()
    except Exception as e:
        print("[DETECT]", e)
        safe_set(status_text,"Camera error, retrying...")
        time.sleep(2)
        if running: detect_loop()  # auto-recover
    finally:
        if cap: cap.release()
        cv2.destroyAllWindows()

# ---------------- Watchdog ----------------
def watchdog():
    """Restart detection loop if camera thread dies silently."""
    global watchdog_alive
    last_time=time.time()
    while watchdog_alive:
        time.sleep(8)
        if not running: continue
        if (time.time()-last_time)>10:
            print("[WATCHDOG] restarting detection")
            threading.Thread(target=detect_loop,daemon=True).start()
        last_time=time.time()

# ---------------- Controls ----------------
def start_app():
    global running, voice_running, watchdog_alive
    if running: return
    running=True; voice_running=True; watchdog_alive=True
    safe_set(status_text,"Status: Starting EyeMate...")
    threading.Thread(target=detect_loop,daemon=True).start()
    threading.Thread(target=voice_listener,daemon=True).start()
    threading.Thread(target=watchdog,daemon=True).start()

def stop_detection():
    global running, watchdog_alive
    running=False; watchdog_alive=False
    try:
        if cap: cap.release()
    except: pass
    cv2.destroyAllWindows()
    safe_set(status_text,"Status: Object detection stopped.")
    safe_set(listening_text,"🎤 Status: Not Listening")

def return_to_idle():
    global running,voice_running,scanning
    running=False; voice_running=False; scanning=False
    safe_set(status_text,"Status: Idle")
    safe_set(listening_text,"🎤 Status: Not Listening")
    safe_set(narration_text,"Narration: ---")

# ---------------- Buttons ----------------
Button(window,text="Start EyeMate",command=start_app,
       bg="#FEE715",fg="#101820",font=("Arial",12,"bold")).pack(pady=10)
Button(window,text="Exit Application",
       command=lambda:[speak_async("EyeMate application closing."),
                       return_to_idle(),window.after(900,window.destroy)],
       bg="#B22222",fg="white",font=("Arial",12,"bold")).pack(pady=5)

# ---------------- Main ----------------
window.mainloop()
