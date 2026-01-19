import time
import cv2
import os
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from roi_functions import (scale_roi, prepare_static_roi, safe_extract_roi)
from helper_functions import CameraLoader, predict_material_batch
from monitor_hook import emit_frame_result

# ============================================================
# CONSTANTS
# ============================================================


os.makedirs("output_frames", exist_ok=True)

# ============================================================
# LOAD MODELS (ONCE PER PROCESS)
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt")
model.to(device)

# ============================================================
# ROI UTILITIES (ALL IN THIS FILE)
# ============================================================

# ============================================================
# MAIN LOOP
# ============================================================

def main(line_number, rtsp_dict, camera_cfg, stop_event):
    """
    Main processing loop for a single production line.
    Detection happens ONLY inside ROI.
    """

    video_capture = CameraLoader(rtsp_dict)
    video_capture.start()

    camera_id = camera_cfg["cameraid"]
    roi_original = camera_cfg["roi"]  # ROI drawn at 1280x720

    print(f"✅ Line {line_number} | Camera {camera_id} starting")

    # --------------------------------------------------------
    # WAIT FOR FIRST FRAME (DETECT RTSP RESOLUTION)
    # --------------------------------------------------------

    # while not video_capture.frame_set:
    #     time.sleep(0.01)

    # first_cam = next(iter(video_capture.frame_set))
    # first_frame = video_capture.frame_set[first_cam][2]

    # if first_frame is None:
    #     raise RuntimeError("❌ Failed to read initial RTSP frame")

    # frame_h, frame_w = first_frame.shape[:2]
    frame_h, frame_w = 360,640
    print(f"📐 RTSP Resolution detected: {frame_w}x{frame_h}")

    # --------------------------------------------------------
    # SCALE ROI ONCE
    # --------------------------------------------------------

    roi_scaled = scale_roi(
        roi_points=roi_original,
        target_size=(frame_w, frame_h)
    )

    roi_bbox, roi_mask, _ = prepare_static_roi(roi_scaled)
    bx, by, bw, bh = roi_bbox

    print(f"📐 ROI bbox after scaling: {roi_bbox}")

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # --------------------------------------------------------

    try:
        while not stop_event.is_set():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            frames_batch = []
            full_frames = []

            # ---------------- COLLECT FRAMES ----------------
            for cam_key, frames in video_capture.frame_set.items():
                if not frames or frames[0] is None:
                    continue

                full_frame = frames[0]
                roi_frame = safe_extract_roi(full_frame, roi_bbox, roi_mask)

                if roi_frame is None:
                    continue

                frames_batch.append(roi_frame)
                full_frames.append(full_frame)

            if not frames_batch:
                time.sleep(0.005)
                continue

            t1 = time.time()

            # ---------------- YOLO (ROI ONLY) ----------------
            yolo_results = model(
                frames_batch,
                imgsz=640,
                conf=0.4,
                classes= [0],  # person class only
                verbose=False,
                device=device
            )

            # ---------------- MATERIAL MODEL ----------------
            try:
                material_results = predict_material_batch(frames_batch)
            except Exception:
                material_results = [(None, 0.0)] * len(frames_batch)

            # ---------------- POST PROCESS ----------------
            for full_frame, yolo_res, (mat_cls, mat_prob) in zip(
                full_frames, yolo_results, material_results
            ):
                annotated = full_frame.copy()

                # -------- YOLO BOXES (SAFE) --------
                if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                    cls_ids = yolo_res.boxes.cls.cpu().numpy()
                    mask = cls_ids == 0  # person only

                    boxes_roi = yolo_res.boxes.xyxy.cpu().numpy()[mask]
                    #boxes_roi = yolo_res.boxes.xyxy.cpu().numpy()
                    boxes_full = boxes_roi.copy()

                    boxes_full[:, [0, 2]] += bx
                    boxes_full[:, [1, 3]] += by

                    annotator = Annotator(annotated)

                    for box in boxes_full:
                        annotator.box_label(
                            box,
                            label="person",
                            color=(0, 255, 0)
                        )

                    annotated = annotator.result()

                # -------- ROI OVERLAY --------
                cv2.polylines(
                    annotated,
                    [np.array(roi_scaled, np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )

                # -------- MATERIAL TEXT --------
                cv2.putText(
                    annotated,
                    f"Material: {mat_cls} ({mat_prob:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                # -------- EMIT RESULTS --------
                emit_frame_result(
                    line_id=line_number,
                    cam_id=camera_id,
                    yolo_res=yolo_res,
                    material=(mat_cls, mat_prob),
                    timestamp=time.time()
                )

                # -------- SAVE FRAME (OPTIONAL) --------
                out_path = f"output_frames/frame_cam{camera_id}_{timestamp}.jpg"
                cv2.imwrite(out_path, annotated)

            t2 = time.time()
            print(f"[LINE {line_number}] ROI YOLO + MATERIAL | {t2 - t1:.3f}s")

        print(f"🛑 Stop event received for line {line_number}")

    except Exception as e:
        print(f"❌ Line {line_number} crashed: {e}")

    finally:
        print(f"🧹 Releasing camera for line {line_number}")
        video_capture.stop()
