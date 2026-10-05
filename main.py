import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# ---------------- CONFIG ----------------
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
AMBULANCE_LABELS = {"ambulance"}
USE_FALLBACK = True

# ---------------- STATE ----------------
lane_images = [None]*4
lane_counts = [0]*4
lane_times = [5]*4
ambulance_flags = [False]*4

signal_state = ["RED"]*4
current_capture = 0
capturing_done = False

signal_order = []
current_signal_index = 0
signal_timer = 0
signal_phase = "GREEN"
YELLOW_TIME = 3

# ---------------- DETECTION ----------------
def detect(img):
    # Run YOLO detection
    results = model(img, conf=0.2, imgsz=960, iou=0.5, verbose=False)

    count = 0
    ambulance = False
    annotated = img.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Filter only vehicle classes
            if label not in VEHICLE_CLASSES:
                continue

            count += 1

            # Ambulance detection (only works with custom model)
            if label == "ambulance":
                ambulance = True

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return count, ambulance, annotated

# ---------------- CAPTURE ----------------
def capture_next():
    global current_capture, capturing_done

    if capturing_done:
        return

    status_label.config(text=f"Capturing Lane {current_capture+1}... Adjust Camera")

    ret, frame = cap.read()
    if not ret:
        return

    count, amb, annotated = detect(frame)

    lane_images[current_capture] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    lane_counts[current_capture] = count
    ambulance_flags[current_capture] = amb

    current_capture += 1

    if current_capture == 4:
        capturing_done = True
        calculate_priority()
    else:
        root.after(10000, capture_next)

# ---------------- PRIORITY ----------------
def calculate_priority():
    global signal_order, lane_times

    total = sum(lane_counts)
    lane_times = [max(5, int((c/total)*30)) if total else 5 for c in lane_counts]

    signal_order.clear()

    # Ambulance override
    for i in range(4):
        if ambulance_flags[i]:
            signal_order.append(i)

    normal = list(np.argsort(lane_counts)[::-1])
    for n in normal:
        if n not in signal_order:
            signal_order.append(n)

    start_signal()

# ---------------- SIGNAL ----------------
def start_signal():
    global signal_timer, signal_phase

    lane = signal_order[current_signal_index]

    for i in range(4):
        signal_state[i] = "RED"

    signal_state[lane] = "GREEN"
    signal_phase = "GREEN"
    signal_timer = lane_times[lane]

def update_signal():
    global signal_timer, current_signal_index, signal_phase

    if not capturing_done:
        root.after(1000, update_signal)
        return

    if signal_timer > 0:
        signal_timer -= 1
    else:
        lane = signal_order[current_signal_index]

        if signal_phase == "GREEN":
            signal_state[lane] = "YELLOW"
            signal_phase = "YELLOW"
            signal_timer = YELLOW_TIME
        else:
            current_signal_index += 1

            if current_signal_index >= 4:
                reset_system()
                return

            start_signal()

    root.after(1000, update_signal)

# ---------------- RESET ----------------
def reset_system():
    global lane_images, lane_counts, lane_times, ambulance_flags
    global signal_state, current_capture, capturing_done
    global signal_order, current_signal_index

    lane_images = [None]*4
    lane_counts = [0]*4
    lane_times = [5]*4
    ambulance_flags = [False]*4

    signal_state = ["RED"]*4
    current_capture = 0
    capturing_done = False
    signal_order = []
    current_signal_index = 0

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Smart Traffic Control System")
root.geometry("1000x750")
root.configure(bg="#0b1e3d")

title = tk.Label(root,
    text="🚦 Smart Traffic Control System",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#0b1e3d")
title.grid(row=0, column=0, columnspan=2, pady=10)

frames = []
image_labels = []
info_labels = []
canvases = []

for i in range(4):
    frame = tk.Frame(root, bg="#132c5a", bd=2, relief="ridge")
    frame.grid(row=(i//2)+1, column=i%2, padx=10, pady=10)

    img_label = tk.Label(frame, bg="#132c5a")
    img_label.pack(side="left")

    info_frame = tk.Frame(frame, bg="#132c5a")
    info_frame.pack(side="left", padx=5)

    info = tk.Label(info_frame,
        text="Vehicles: 0\nEmergency: No\nSignal: RED\nTimer: 0s",
        fg="white", bg="#132c5a", justify="left")
    info.pack()

    canvas = tk.Canvas(info_frame, width=80, height=200, bg="black", highlightthickness=0)
    canvas.pack(pady=5)

    r = canvas.create_oval(20,10,60,50, fill="grey")
    y = canvas.create_oval(20,70,60,110, fill="grey")
    g = canvas.create_oval(20,130,60,170, fill="grey")

    frames.append(frame)
    image_labels.append(img_label)
    info_labels.append(info)
    canvases.append((canvas,r,y,g))

status_label = tk.Label(root, text="Press Start to Capture",
    fg="white", bg="#0b1e3d", font=("Arial", 12))
status_label.grid(row=4, column=0, columnspan=2)

btn = ttk.Button(root, text="Start Capture", command=capture_next)
btn.grid(row=5, column=0, columnspan=2, pady=10)

# ---------------- GUI UPDATE ----------------
def update_gui():
    ret, live_frame = cap.read()
    if not ret:
        root.after(100, update_gui)
        return

    live_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)

    for i in range(4):
        if lane_images[i] is not None:
            img = lane_images[i]
        else:
            img = live_frame

        img = cv2.resize(img, (350,200))
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        image_labels[i].imgtk = imgtk
        image_labels[i].config(image=imgtk)

        canvas,r,y,g = canvases[i]
        canvas.itemconfig(r, fill="red" if signal_state[i]=="RED" else "grey")
        canvas.itemconfig(y, fill="yellow" if signal_state[i]=="YELLOW" else "grey")
        canvas.itemconfig(g, fill="green" if signal_state[i]=="GREEN" else "grey")

        text = f"Vehicles: {lane_counts[i]}\n"
        # text += f"Emergency: {'Yes' if ambulance_flags[i] else 'No'}\n"
        text += f"Signal: {signal_state[i]}\n"
        text += f"Timer: {signal_timer}s" if signal_state[i]=="GREEN" else "Timer: --"

        info_labels[i].config(text=text)

    root.after(100, update_gui)

update_gui()
update_signal()
root.mainloop()