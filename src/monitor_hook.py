from monitoring_engine import process_frame

def emit_frame_result(line_id, cam_id, yolo_res, material, timestamp):
    print("monitoring on line ID", line_id)
    process_frame(line_id, cam_id, yolo_res, material, timestamp)