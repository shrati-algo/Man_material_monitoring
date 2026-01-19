
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import deque
import os
import cv2
import time
from threading import Thread, Lock
from queue import Queue, Empty
# ---------------- FFMPEG OPTIONS ----------------


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay"
)




# class CameraLoader:
#     def __init__(self, config_setting):
#         """
#         config_setting format:
#         {
#             "cam1": {"rtsp": "rtsp://...", "cameraid": 1},
#             "cam2": {"rtsp": "rtsp://...", "cameraid": 2}
#         }
#         """
#         self.config_setting = config_setting
#         self.video_objects = {}
#         self.frame_set = {}
#         self.threads = {}
#         self.stopped = False

#         # -------- REGISTER ALL CAMERAS (EVEN IF DOWN) --------
#         for cam_name, cam_cfg in self.config_setting.items():
#             rtsp = cam_cfg.get("rtsp")
#             cam_id = cam_cfg.get("cameraid")

#             if not rtsp or cam_id is None:
#                 print(f"❌ Invalid config for {cam_name}")
#                 continue

#             cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

#             self.video_objects[cam_id] = {
#                 "name": cam_name,
#                 "cameraID": cam_id,
#                 "rtsp": rtsp,
#                 "capture_obj": cap,
#                 "failed_at_startup": not cap.isOpened()  # ✅ Track startup failures
#             }

#             self.frame_set[cam_id] = deque(maxlen=1)

#             if cap.isOpened():
#                 print(f"✅ Camera {cam_id} ({cam_name}) opened")
#             else:
#                 print(f"⚠️ Camera {cam_id} ({cam_name}) DOWN at startup, will retry in background")

#         print(f"✅ CameraLoader initialized with {len(self.video_objects)} cameras")

#     # ---------------- START ALL CAMERA THREADS ----------------
#     def start(self):
#         for cam_id in self.video_objects:
#             t = Thread(target=self._reader, args=(cam_id,), daemon=True)
#             t.start()
#             self.threads[cam_id] = t
#             time.sleep(0.3)  # stagger startup

#         return self

#     # ---------------- READER THREAD WITH RETRY LOGIC ----------------
#     def _reader(self, cam_id):
#         cam = self.video_objects[cam_id]
#         rtsp = cam["rtsp"]
#         cap = cam["capture_obj"]

#         MAX_RETRIES = 3
#         NO_FRAME_TIMEOUT = 3

#         retry_count = 0
#         last_reconnect_time = 0
#         permanently_disabled = False

#         # ✅ Handle cameras that failed at startup - IMMEDIATE first reconnect
#         if cam.get("failed_at_startup", False):
#             print(f"🔄 Camera {cam_id} attempting reconnect 1/{MAX_RETRIES} (startup failure)")
#             retry_count = 1
#             cap.release()
#             time.sleep(1)  # Brief delay before first retry
#             cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
#             cam["capture_obj"] = cap
#             last_reconnect_time = time.time()

#         while not self.stopped:
#             now = time.time()

#             # ✅ Check if permanently disabled
#             if permanently_disabled:
#                 time.sleep(5)
#                 continue

#             # ✅ Try to read frame
#             try:
#                 ret, frame = cap.read()
#             except cv2.error as e:
#                 retry_count += 1
#                 if retry_count > MAX_RETRIES:
#                     print(f"❌ Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (OpenCV error)")
#                     permanently_disabled = True
#                     cap.release()
#                     continue

#                 print(f"❌ Camera {cam_id} OpenCV error → reconnect attempt {retry_count}/{MAX_RETRIES}")
#                 cap.release()
#                 time.sleep(2)
#                 cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
#                 cam["capture_obj"] = cap
#                 last_reconnect_time = time.time()
#                 continue
#             except Exception as e:
#                 retry_count += 1
#                 if retry_count > MAX_RETRIES:
#                     print(f"❌ Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (exception: {str(e)[:50]})")
#                     permanently_disabled = True
#                     cap.release()
#                     continue

#                 print(f"❌ Camera {cam_id} exception → reconnect attempt {retry_count}/{MAX_RETRIES}")
#                 cap.release()
#                 time.sleep(2)
#                 cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
#                 cam["capture_obj"] = cap
#                 last_reconnect_time = time.time()
#                 continue

#             # ✅ Handle failed frame reads
#             if not ret or frame is None:
#                 time_since_reconnect = now - last_reconnect_time

#                 # ✅ Trigger reconnect after timeout
#                 if time_since_reconnect >= NO_FRAME_TIMEOUT:
#                     retry_count += 1

#                     if retry_count > MAX_RETRIES:
#                         print(f"❌ Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (no frames)")
#                         permanently_disabled = True
#                         cap.release()
#                         continue

#                     print(f"⚠️ Camera {cam_id} no frames for {NO_FRAME_TIMEOUT}s → reconnect attempt {retry_count}/{MAX_RETRIES}")
#                     cap.release()
#                     time.sleep(2)
#                     cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
#                     cam["capture_obj"] = cap
#                     last_reconnect_time = time.time()
#                 else:
#                     time.sleep(0.2)
#                 continue

#             # ✅ SUCCESS PATH - frame received
#             if retry_count > 0:
#                 print(f"✅ Camera {cam_id} reconnected successfully after {retry_count} attempts")

#             retry_count = 0
#             last_reconnect_time = time.time()
#             self.frame_set[cam_id].append(frame)

#     # ---------------- UTILITY METHODS ----------------
#     def get_active_cameras(self):
#         """Returns list of camera IDs that have frames available"""
#         return [cid for cid, frames in self.frame_set.items() if len(frames) > 0]

#     def get_frame(self, cam_id):
#         """Get the latest frame for a specific camera"""
#         if cam_id in self.frame_set and len(self.frame_set[cam_id]) > 0:
#             return self.frame_set[cam_id][0]
#         return None

#     def is_camera_active(self, cam_id):
#         """Check if a specific camera is currently active"""
#         return cam_id in self.frame_set and len(self.frame_set[cam_id]) > 0

#     # ---------------- STOP ALL ----------------
#     def stop(self):
#         """Stop all camera threads and release resources"""
#         print("🛑 Stopping CameraLoader...")
#         self.stopped = True

#         # Wait for threads to finish
#         for cam_id, thread in self.threads.items():
#             if thread.is_alive():
#                 thread.join(timeout=2)

#         # Release all video captures
#         for cam in self.video_objects.values():
#             try:
#                 cam["capture_obj"].release()
#             except Exception as e:
#                 pass

#         print("✅ CameraLoader stopped")






class CameraLoader:
    def __init__(self, config_setting, num_threads=3):
        """
        config_setting format:
        {
            "cam1": {"rtsp": "rtsp://...", "cameraid": 1},
            "cam2": {"rtsp": "rtsp://...", "cameraid": 2}
        }
        num_threads: Number of worker threads to use (default: 4)
        """
        self.config_setting = config_setting
        self.num_threads = num_threads
        self.video_objects = {}
        self.frame_set = {}
        self.stopped = False
        self.lock = Lock()

        # Task queue for thread pool
        self.task_queue = Queue()
        self.worker_threads = []

        # -------- REGISTER ALL CAMERAS (EVEN IF DOWN) --------
        for cam_name, cam_cfg in self.config_setting.items():
            rtsp = cam_cfg.get("rtsp")
            cam_id = cam_cfg.get("cameraid")

            if not rtsp or cam_id is None:
                print(f"❌ Invalid config for {cam_name}")
                continue

            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

            self.video_objects[cam_id] = {
                "name": cam_name,
                "cameraID": cam_id,
                "rtsp": rtsp,
                "capture_obj": cap,
                "failed_at_startup": not cap.isOpened(),
                "retry_count": 0,
                "last_reconnect_time": 0,
                "permanently_disabled": False
            }

            self.frame_set[cam_id] = deque(maxlen=1)

            if cap.isOpened():
                print(f"✅ Camera {cam_id} ({cam_name}) opened")
            else:
                print(f"⚠️ Camera {cam_id} ({cam_name}) DOWN at startup, will retry in background")

        print(f"✅ CameraLoader initialized with {len(self.video_objects)} cameras, {num_threads} worker threads")

    # ---------------- START THREAD POOL ----------------
    def start(self):
        """Start worker threads and populate task queue"""
        # Start worker threads
        for i in range(self.num_threads):
            t = Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.worker_threads.append(t)

        # Populate task queue with all camera IDs
        for cam_id in self.video_objects.keys():
            self.task_queue.put(cam_id)

        print(f"✅ Started {self.num_threads} worker threads processing {len(self.video_objects)} cameras")
        return self

    # ---------------- WORKER THREAD ----------------
    def _worker(self, worker_id):
        """Worker thread that processes cameras from the queue"""
        print(f"🔧 Worker {worker_id} started")

        while not self.stopped:
            try:
                # Get a camera to process (timeout to check stopped flag)
                cam_id = self.task_queue.get(timeout=1)
            except Empty:
                continue

            # Process this camera (read one frame)
            self._process_camera(cam_id, worker_id)

            # Put the camera back in the queue for continuous processing
            if not self.stopped:
                self.task_queue.put(cam_id)

            self.task_queue.task_done()

    # ---------------- PROCESS SINGLE CAMERA ----------------
    def _process_camera(self, cam_id, worker_id):
        """Process a single frame read for a camera"""
        MAX_RETRIES = 3
        NO_FRAME_TIMEOUT = 3

        with self.lock:
            cam = self.video_objects[cam_id]

            # Skip if permanently disabled
            if cam["permanently_disabled"]:
                time.sleep(5)
                return

            cap = cam["capture_obj"]
            retry_count = cam["retry_count"]
            last_reconnect_time = cam["last_reconnect_time"]

        # ✅ Handle cameras that failed at startup - IMMEDIATE first reconnect
        if cam.get("failed_at_startup", False):
            with self.lock:
                if cam["retry_count"] == 0:  # Only do this once
                    print(f"🔄 [Worker {worker_id}] Camera {cam_id} attempting reconnect 1/{MAX_RETRIES} (startup failure)")
                    cam["retry_count"] = 1
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(cam["rtsp"], cv2.CAP_FFMPEG)
                    cam["capture_obj"] = cap
                    cam["last_reconnect_time"] = time.time()
                    cam["failed_at_startup"] = False
                    return

        now = time.time()

        # ✅ Try to read frame
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            with self.lock:
                cam["retry_count"] += 1
                if cam["retry_count"] > MAX_RETRIES:
                    print(f"❌ [Worker {worker_id}] Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (OpenCV error)")
                    cam["permanently_disabled"] = True
                    cap.release()
                    return

                print(f"❌ [Worker {worker_id}] Camera {cam_id} OpenCV error → reconnect attempt {cam['retry_count']}/{MAX_RETRIES}")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam["rtsp"], cv2.CAP_FFMPEG)
                cam["capture_obj"] = cap
                cam["last_reconnect_time"] = time.time()
            return
        except Exception as e:
            with self.lock:
                cam["retry_count"] += 1
                if cam["retry_count"] > MAX_RETRIES:
                    print(f"❌ [Worker {worker_id}] Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (exception: {str(e)[:50]})")
                    cam["permanently_disabled"] = True
                    cap.release()
                    return

                print(f"❌ [Worker {worker_id}] Camera {cam_id} exception → reconnect attempt {cam['retry_count']}/{MAX_RETRIES}")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam["rtsp"], cv2.CAP_FFMPEG)
                cam["capture_obj"] = cap
                cam["last_reconnect_time"] = time.time()
            return

        # ✅ Handle failed frame reads
        if not ret or frame is None:
            time_since_reconnect = now - cam["last_reconnect_time"]

            # ✅ Trigger reconnect after timeout
            if time_since_reconnect >= NO_FRAME_TIMEOUT:
                with self.lock:
                    cam["retry_count"] += 1

                    if cam["retry_count"] > MAX_RETRIES:
                        print(f"❌ [Worker {worker_id}] Camera {cam_id} permanently disabled after {MAX_RETRIES} retries (no frames)")
                        cam["permanently_disabled"] = True
                        cap.release()
                        return

                    print(f"⚠️ [Worker {worker_id}] Camera {cam_id} no frames for {NO_FRAME_TIMEOUT}s → reconnect attempt {cam['retry_count']}/{MAX_RETRIES}")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(cam["rtsp"], cv2.CAP_FFMPEG)
                    cam["capture_obj"] = cap
                    cam["last_reconnect_time"] = time.time()
            else:
                time.sleep(0.2)
            return

        # ✅ SUCCESS PATH - frame received
        with self.lock:
            if cam["retry_count"] > 0:
                print(f"✅ [Worker {worker_id}] Camera {cam_id} reconnected successfully after {cam['retry_count']} attempts")

            cam["retry_count"] = 0
            cam["last_reconnect_time"] = time.time()
            self.frame_set[cam_id].append(frame)

    # ---------------- UTILITY METHODS ----------------
    def get_active_cameras(self):
        """Returns list of camera IDs that have frames available"""
        with self.lock:
            return [cid for cid, frames in self.frame_set.items() if len(frames) > 0]

    def get_frame(self, cam_id):
        """Get the latest frame for a specific camera"""
        with self.lock:
            if cam_id in self.frame_set and len(self.frame_set[cam_id]) > 0:
                return self.frame_set[cam_id][0]
        return None

    def is_camera_active(self, cam_id):
        """Check if a specific camera is currently active"""
        with self.lock:
            return cam_id in self.frame_set and len(self.frame_set[cam_id]) > 0

    # ---------------- STOP ALL ----------------
    def stop(self):
        """Stop all worker threads and release resources"""
        print("🛑 Stopping CameraLoader...")
        self.stopped = True

        # Wait for workers to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2)

        # Release all video captures
        with self.lock:
            for cam in self.video_objects.values():
                try:
                    cam["capture_obj"].release()
                except Exception as e:
                    pass

        print("✅ CameraLoader stopped")




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MATERIAL MODEL --------
material_model = models.efficientnet_b0(weights=None)
material_model.classifier[1] = torch.nn.Linear(
    material_model.classifier[1].in_features, 1
)

material_model.load_state_dict(
    torch.load(
        r"src\models\material_detector.pth",
        map_location=DEVICE,
        weights_only=True
    )
)


material_model.eval().to(DEVICE)

material_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def predict_material_batch(frames_bgr, threshold=0.5):
    """
    frames_bgr: list of OpenCV BGR images
    returns: list of (cls, prob)
    """
    tensors = []

    for frame in frames_bgr:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensors.append(material_tfms(img))

    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(material_model(batch)).squeeze(1)

    return [(int(p >= threshold), float(p)) for p in probs]

 