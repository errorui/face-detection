import requests
import time
from PIL import Image
import numpy as np


API_URL = "http://localhost:8000/recognize-face-rgb565/"
IMAGE_PATH = r"sample_person/WhatsApp Image 2025-04-11 at 7.22.07 PM.jpeg"
def convert_image_to_raw_rgb565(image_path: str):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    height, width = arr.shape[:2]

    rgb565 = []
    for y in range(height):
        for x in range(width):
            r, g, b = arr[y, x]
            color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            rgb565.append(color)

    return width, height, bytes([(v >> 8) & 0xFF for v in rgb565] + [v & 0xFF for v in rgb565])

def send_rgb565_to_api(image_path):
    width, height, rgb565_bytes = convert_image_to_raw_rgb565(image_path)

    files = {
        "file": ("image.rgb565", rgb565_bytes, "application/octet-stream")
    }
    url_with_params = f"{API_URL}?width={width}&height={height}"

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(url_with_params, files=files)
            if response.status_code == 200:
                print(f"✅ Prediction for {image_path}: {response.json()}")
                return response.json()
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(2)

    print("❌ Request failed after retries.")

# Run the test
send_rgb565_to_api(IMAGE_PATH)
