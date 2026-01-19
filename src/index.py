from flask import Flask, request, jsonify
import multiprocessing as mp
import src.capture as capture
from config_loader import get_camera_config_by_line, build_rtsp_dict

app = Flask(__name__)

# ---------------- STATE ----------------
running_lines = {}
stop_flags = {}    


# ---------------- START LINE ----------------
@app.route("/start", methods=["POST"])
def start_line():
    data = request.get_json()

    if not data or "line_number" not in data:
        return jsonify({"error": "line_number is required"}), 400

    line_number = data["line_number"]

    if line_number in running_lines:
        return jsonify({"error": f"Line {line_number} already running"}), 400

    try:
        camera_cfg = get_camera_config_by_line(line_number)
        rtsp_dict = build_rtsp_dict(camera_cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    stop_event = mp.Event()
    stop_flags[line_number] = stop_event

    p = mp.Process(
        target=capture.main,
        args=(line_number, rtsp_dict, camera_cfg, stop_event),
        daemon=True
    )

    p.start()
    running_lines[line_number] = p

    return jsonify({
        "status": "started",
        "line_number": line_number,
        "pid": p.pid,
        "camera_id": camera_cfg["cameraid"]
    })


# ---------------- STOP LINE ----------------
@app.route("/stop", methods=["POST"])
def stop_line():
    data = request.get_json()

    if not data or "line_number" not in data:
        return jsonify({"error": "line_number is required"}), 400

    line_number = data["line_number"]

    if line_number not in running_lines:
        return jsonify({"error": f"Line {line_number} not running"}), 400

    stop_flags[line_number].set()

    proc = running_lines[line_number]
    proc.join(timeout=10)

    if proc.is_alive():
        proc.terminate()  # HARD KILL fallback

    del running_lines[line_number]
    del stop_flags[line_number]

    return jsonify({
        "status": "stopped",
        "line_number": line_number
    })


# ---------------- HEALTH ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "running_lines": {
            ln: proc.pid for ln, proc in running_lines.items()
        }
    })


# ---------------- MAIN ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # IMPORTANT for CUDA
    app.run(host="0.0.0.0", port=8000, debug=False)
