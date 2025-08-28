🚦 AI-Powered Traffic & Pedestrian Analytics 🚦
Built a Traffic Analytics System that detects, tracks, and estimates vehicle speed from CCTV — no sensors needed !

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/37c8580b-96cb-4528-a432-7abebdcb2729" />

🔹 Objective
 I built a Computer Vision system that:
 ✅ Detects vehicles & people in CCTV/video footage.
 ✅ Tracks them in real-time with unique IDs.
 ✅ Estimates vehicle speed (km/hr).
 ✅ Counts cars, bikes, buses, autos.
 ✅ Performs traffic density analysis:
🚗 < 20 km/hr → High Traffic
🚙 20–25 km/hr → Medium Traffic
🛣 > 25 km/hr → Low Traffic
 ✅ Saves results to CSV logs for business insights.

🔹 Tech Stack & Tools
YOLOv8 (Ultralytics) → Object detection (trained on COCO dataset).
DeepSORT → Multi-object tracking with unique IDs.
OpenCV → Video processing & visualization.
Pandas → Structured CSV export.
Python 🚀

🔹 How it works
 1️⃣ YOLOv8 detects vehicles/people frame by frame.
 2️⃣ DeepSORT assigns unique IDs → keeps tracking across frames.
 3️⃣ Using pixel-to-meter calibration → system calculates speed.
 4️⃣ Real-time vehicle count & category breakdown (car, bus, bike, auto).
 5️⃣ Traffic flow analyzed → generates CSV reports (speed, density, count).

🔹 Potential Use Cases
 📍 Smart Cities – Automated traffic monitoring & congestion alerts.
 📍 Highway Patrol/Traffic Police – Over-speeding detection.
 📍 Transport Analytics – Vehicle type & density analysis.
 📍 Parking Management – Count & track vehicles entering/exiting.
 📍 Retail & Public Spaces – People flow monitoring.
