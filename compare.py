import time
import base64
import requests
import subprocess
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError

IMAGE = "image.jpeg"
URL = "http://localhost:11434/api/generate"
MODEL = "moondream"
PROMPT = "What is happening in this image?"
RUNS = 3


def send_bytes(b):
    payload = {"model": MODEL, "prompt": PROMPT, "images": [base64.b64encode(b).decode()], "stream": False}
    t0 = time.time()
    try:
        r = requests.post(URL, json=payload, timeout=120)
        dt = time.time() - t0
        return r.status_code, r.text, dt
    except Exception as e:
        return None, str(e), time.time() - t0


def optimized_bytes():
    # Try PIL conversion + resize first
    try:
        img = Image.open(IMAGE)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((384, 288), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return buf.getvalue()
    except UnidentifiedImageError:
        # fallback to ffmpeg
        tmp = "/tmp/compare_conv.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", IMAGE, tmp], check=True)
        with open(tmp, "rb") as f:
            data = f.read()
        # ensure resized
        try:
            img = Image.open(BytesIO(data))
            img.thumbnail((384, 288), Image.Resampling.LANCZOS)
            buf = BytesIO(); img.save(buf, format="JPEG", quality=65)
            return buf.getvalue()
        except Exception:
            return data


def baseline_run():
    with open(IMAGE, "rb") as f:
        data = f.read()
    return send_bytes(data)


def optimized_run():
    b = optimized_bytes()
    return send_bytes(b)


def run_benchmark():
    print("Running benchmark (N=%d)" % RUNS)

    print("\nBaseline (raw file) runs:")
    base_times = []
    for i in range(RUNS):
        status, text, dt = baseline_run()
        base_times.append(dt)
        print(f" {i+1}. status={status} time={dt:.3f}s resp_len={len(text) if text else 0}")

    print("\nOptimized (convert+resize) runs:")
    opt_times = []
    for i in range(RUNS):
        status, text, dt = optimized_run()
        opt_times.append(dt)
        print(f" {i+1}. status={status} time={dt:.3f}s resp_len={len(text) if text else 0}")

    import statistics
    print("\nSummary:")
    print(f" Baseline avg: {statistics.mean(base_times):.3f}s (N={RUNS})")
    print(f" Optimized avg: {statistics.mean(opt_times):.3f}s (N={RUNS})")
    print(f" Delta: {statistics.mean(opt_times)-statistics.mean(base_times):+.3f}s")


if __name__ == '__main__':
    if not os.path.exists(IMAGE):
        print(f"Image not found: {IMAGE}")
    else:
        run_benchmark()
