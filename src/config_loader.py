import configparser

config = configparser.ConfigParser()
config_path = "config\config.ini"
config.read(config_path)


def parse_roi(roi_str):
    """
    Converts:
    '406,183|162,719|526,717|625,206'
    → [(406,183), (162,719), (526,717), (625,206)]
    """
    points = []
    for pair in roi_str.split("|"):
        x, y = pair.split(",")
        points.append((int(x), int(y)))
    return points


def get_camera_config_by_line(line_number, config_path="config\config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)

    for section in config.sections():
        if section.startswith(f"PL_{line_number}_CAMERA_"):
            camera_id = int(section.split("_")[-1])

            return {
                "camera_key": f"cam{camera_id}",
                "cameraid": camera_id,
                "rtsp": config.get(section, "rtsp"),
                "roi": parse_roi(config.get(section, "roi")),
                "min_person_count": config.getint(section, "min_person_count"),
                "person_absent_duration_sec": config.getint(section, "person_absent_duration_sec"),
                "material_absent_duration_sec": config.getint(section, "material_absent_duration_sec"),
            }

    raise ValueError(f"No camera found for line {line_number}")

def build_rtsp_dict(camera_cfg):
    return {
        camera_cfg["camera_key"]: {
            "rtsp": camera_cfg["rtsp"],
            "cameraid": camera_cfg["cameraid"]
        }
    }

def get_line_thresholds(line_number, config_path="config\config.ini"):

    thresholds = {}
    for section in config.sections():
        if section.startswith(f"PL_{line_number}_CAMERA_"):
            camera_id = int(section.split("_")[-1])
            thresholds[camera_id] = {
                "min_person_count": config.getint(section, "min_person_count"),
                "person_absent_duration_sec": config.getint(section, "person_absent_duration_sec"),
                "material_absent_duration_sec": config.getint(section, "material_absent_duration_sec"),
            }

    if not thresholds:
        raise ValueError(f"No thresholds found for line {line_number}")

    return thresholds