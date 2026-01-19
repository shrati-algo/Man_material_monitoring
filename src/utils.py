
import time
from datetime import datetime

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} took {end - start:.2f}s")
        return result
    return wrapper



def log(msg, cam_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[CAM {cam_id}]" if cam_id is not None else "[INFO]"
    print(f"{ts} {prefix} {msg}")
