import time
from collections import defaultdict
from violation_handler import raise_violation
from config_loader import get_line_thresholds
# ---------------- STATE ----------------
last_seen_time = defaultdict(lambda: time.time())


def process_frame(
    line_id,
    cam_id,
    yolo_res,
    material,
    timestamp
):
    """
    cam_id  : camera id (int or string)
    yolo_res: YOLO result object
    material: (mat_cls, mat_prob)
    timestamp: time.time()
    thresholds: passed from config
    """
    min_person_count = get_line_thresholds(line_id)[cam_id]["min_person_count"]
    person_absent_duration_sec= get_line_thresholds(line_id)[cam_id]["person_absent_duration_sec"]
    material_absent_duration_sec= get_line_thresholds(line_id)[cam_id]["material_absent_duration_sec"]
    
    # -------- PERSON COUNT --------
    person_count = 0
    if yolo_res.boxes is not None:
        person_count = sum(
            int(cls == 0) for cls in yolo_res.boxes.cls.tolist()
        )

    # -------- MATERIAL PRESENCE --------
    mat_cls, mat_prob = material
    material_present = mat_prob > 0.5

    # -------- UPDATE LAST SEEN --------
    if person_count > 0 or material_present:
        last_seen_time[cam_id] = timestamp

    # -------- RULE 1: LOW MAN COUNT --------
    if person_count < min_person_count:
        raise_violation(
            cam_id,
            line_id,
            "LOW_MAN_COUNT",
            f"Count={person_count}, Min={min_person_count}"
        )

    # -------- RULE 2: PERSON ABSENT --------
    elapsed = timestamp - last_seen_time[cam_id]
    if elapsed > person_absent_duration_sec:
        raise_violation(
            cam_id,
            line_id,
            "PERSON_ABSENT",
            f"Absent for {elapsed/60:.1f} mins"
        )

    # -------- RULE 3: MATERIAL ABSENT --------
    if not material_present and elapsed > material_absent_duration_sec:
        raise_violation(
            cam_id,
            line_id,
            "MATERIAL_ABSENT",
            f"Absent for {elapsed/60:.1f} mins"
        )
