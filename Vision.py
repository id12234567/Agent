import base64
import os
import requests
import time
import cv2
import numpy as np

CV2_AVAILABLE = True
ESP32_URL = "http://192.168.4.1"
IMAGE_PATH = "image.jpeg"
URL = "http://localhost:11434/api/generate"
VISION_MODEL = "moondream"
SUMMARY_MODEL = "llama3.2"

def get_snapshot_bytes():
    try:
        response = requests.get(f"{ESP32_URL}/capture", timeout=10)
        if response.status_code == 200:
            return response.content
    except Exception:
        pass
    return None

def summarize(previous, current):
    prompt = "list the added and removed objects. Keep short & precise. No explaination."
    prompt += "\n\n"
    prompt += f"Previous list of objects: {previous}\n"
    prompt += f"Current list of objects : {current}\n"    
    
    try:
        r = requests.post(
            URL,
            json={
                "model": SUMMARY_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        if r.status_code != 200:
            return ""
        body = r.json()
        summary = (body.get("response") or (body.get("message") or {}).get("content") or "").strip()
        return summary
    except Exception:
        return ""


def main():
    display_enabled = True
    diff = ""
    prev = ""
    print("starting...")

    cv2.namedWindow("Vision", cv2.WINDOW_AUTOSIZE)
    print("OpenCV window initialized: Vision")

    while True:
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)
            
        print("capturing image...")
        image_bytes = get_snapshot_bytes()
        if not image_bytes:
            continue
        with open(IMAGE_PATH, "wb") as f:
            f.write(image_bytes)

        nparr = np.frombuffer(image_bytes, dtype="uint8")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imshow("Vision", image)
            cv2.waitKey(1)

        payload = {
            "model": VISION_MODEL,
            "prompt": "What house hold objects do you see in the picture. Output list of objects only. No sentences. No explaination",
            "images": [base64.b64encode(image_bytes).decode("utf-8")],
            "stream": False,
        }

        
        print("using vision...")
        r = requests.post(URL, json=payload, timeout=60)
        if r.status_code != 200:
            current = "error"
            continue
        else:
            body = r.json()
            current = body.get("response") or (body.get("message") or {}).get("content") or ""
            current = current.strip() or "empty"
            print(f"current: {current}")

        
        print("finding differences...")
        diff = summarize(prev, current)        
        print(f"difference: {diff}")
        prev = current


if __name__ == "__main__":
    main()
