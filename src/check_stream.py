import cv2
import time

# ----------------------------------------
# Stream URL (Flask camera stream)
# ----------------------------------------
STREAM_URL = "http://localhost:5001/cam1"
OUTPUT_VIDEO = r"output_cam1.mp4"

# ----------------------------------------
# Open Stream
# ----------------------------------------
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open stream")

# ----------------------------------------
# Get stream properties
# ----------------------------------------
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Flask MJPEG streams often report FPS=0
if fps == 0 or fps is None:
    fps = 25  # fallback

# ----------------------------------------
# Video Writer
# ----------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"Saving stream to {OUTPUT_VIDEO}")

# ----------------------------------------
# Read & Save Loop
# ----------------------------------------
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame dropped")
        time.sleep(0.01)
        continue

    out.write(frame)

    # Optional: stop after 30 seconds
    if time.time() - start > 30:
        break

# ----------------------------------------
# Cleanup
# ----------------------------------------
cap.release()
out.release()

print("✅ Video saved successfully")
