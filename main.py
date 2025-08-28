# =========================
# YOLOv8 + DeepSORT + Speed + Analytics
# =========================
!pip install ultralytics deep-sort-realtime pandas

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = " "
OUT_PATH   = " "    
WEIGHTS    = "yolov8n.pt"
IMG_SIZE   = 640
CONF_THRES = 0.25
PX_PER_METER = 100   

model = YOLO(WEIGHTS)
tracker = DeepSort(max_age=30)

CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",      
    1: "bicycle",
    0: "person",
}


vehicle_counts = {"car":0, "motorcycle":0, "bus":0, "truck":0, "bicycle":0}
analytics_data = []   # for CSV export

def compute_speed(p1, p2, dt, px_per_meter=20):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    pix = (dx*dx + dy*dy)**0.5
    meters = pix / px_per_meter
    mps = meters / dt
    return mps * 3.6   # km/h

def traffic_level(speed):
    if speed is None:
        return "Unknown"
    if speed < 20: return "High Traffic"
    elif 20 <= speed <= 25: return "Medium Traffic"
    else: return "Low Traffic"


cap = cv2.VideoCapture(VIDEO_PATH)
W, H = int(cap.get(3)), int(cap.get(4))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), FPS, (W,H))

last_pos, last_time = {}, {}

while True:
    ret, frame = cap.read()
    if not ret: break
    now_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        detections.append(([x1,y1,x2-x1,y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)
    for t in tracks:
        if not t.is_confirmed(): continue
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        tid = t.track_id
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cls = t.det_class if hasattr(t, "det_class") else None
        cls_name = CLASS_MAP.get(cls, "other")

       
        if tid not in last_pos and cls_name in vehicle_counts:
            vehicle_counts[cls_name] += 1

       
        speed_kph = None
        if tid in last_pos:
            dt_s = (now_ms - last_time[tid]) / 1000.0
            if dt_s > 0:
                speed_kph = compute_speed(last_pos[tid], (cx,cy), dt_s, PX_PER_METER)

        last_pos[tid] = (cx,cy)
        last_time[tid] = now_ms

        traffic = traffic_level(speed_kph)


        analytics_data.append({
            "track_id": tid,
            "class": cls_name,
            "speed(km/h)": round(speed_kph,2) if speed_kph else None,
            "traffic_level": traffic,
            "time(ms)": now_ms
        })


        color = (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"ID {tid} {cls_name}"
        if speed_kph: label += f" | {speed_kph:.1f} km/h"
        label += f" | {traffic}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1, cv2.LINE_AA)

    writer.write(frame)

cap.release()
writer.release()

# ---- Save CSV ----
df = pd.DataFrame(analytics_data)
df.to_csv("/content/traffic_analytics.csv", index=False)

print("✅ Done! Video saved to:", OUT_PATH)
print("✅ CSV saved to: /content/traffic_analytics.csv")
print("Vehicle Counts:", vehicle_counts)
