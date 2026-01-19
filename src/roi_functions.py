import numpy as np
import cv2

# ============================================================
# ROI REFERENCE RESOLUTION (STATIC FOR ALL CAMERAS)
# ============================================================

ROI_REF_WIDTH = 1280
ROI_REF_HEIGHT = 720

# ============================================================
# ROI SCALING
# ============================================================

def scale_roi(roi_points, target_size):
    """
    Scales ROI polygon from 1280x720 reference resolution
    to actual RTSP frame size.

    roi_points : [(x,y), ...] defined at 1280x720
    target_size: (width, height) of RTSP frame
    """
    tgt_w, tgt_h = target_size

    sx = tgt_w / ROI_REF_WIDTH
    sy = tgt_h / ROI_REF_HEIGHT

    return [(int(x * sx), int(y * sy)) for x, y in roi_points]

# ============================================================
# ROI PREPARATION (BBOX + MASK)
# ============================================================

def prepare_static_roi(roi_points):
    """
    Prepares:
    - ROI bounding box (x, y, w, h)
    - ROI mask (bbox-sized)
    - Shifted polygon (for debugging/overlay if needed)

    This is SAFE for ROI-only inference pipelines.
    """
    pts = np.array(roi_points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(pts)
    roi_bbox = (x, y, w, h)

    shifted_pts = pts - [x, y]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_pts], 255)

    return roi_bbox, mask, shifted_pts

# ============================================================
# ROI EXTRACTION (FAST & SAFE)
# ============================================================

def safe_extract_roi(frame, roi_bbox, roi_mask):
    """
    Extracts masked ROI safely from full frame.

    - Expects bbox-sized mask (NOT full-frame mask)
    - Does NOT resize masks
    - Guards against out-of-frame access
    - Compatible with your existing pipeline
    """
    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    x, y, w, h = roi_bbox
    fh, fw = frame.shape[:2]

    # Bounds check
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        return None

    roi_crop = frame[y:y + h, x:x + w]

    try:
        roi_masked = cv2.bitwise_and(roi_crop, roi_crop, mask=roi_mask)
    except cv2.error as e:
        print(f"⚠️ OpenCV ROI extraction error: {e}")
        return None

    return roi_masked

# ============================================================
# OPTIONAL: FULL-FRAME POLYGON CROP (NOT USED BY YOLO PIPELINE)
# ============================================================

def crop_polygon_roi(frame, roi_points):
    """
    Full-frame polygon crop.
    Use ONLY if you need absolute masking, not ROI inference.
    """
    if frame is None:
        return None, None

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(roi_points, np.int32)

    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked[y:y + h, x:x + w]

    return cropped, (x, y)
