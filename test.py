import requests
import time
from PIL import Image
import numpy as np

url = "http://localhost:8000/recognize-face/"
url2="http://localhost:8000/recognize-face-rgb565/"
# Image path
image_path = r"sample_person/WhatsApp Image 2025-04-11 at 5.38.18 PM (1).jpeg"

# Convert and return RGB565 byte data
# Convert and return RGB565 byte data
def convert_image_to_rgb565(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_array = np.array(img)
        # Convert RGB888 to RGB565: RGB565 uses 16 bits (2 bytes) per pixel
        rgb565 = ((img_array[..., 0] >> 3) << 11) | \
                 ((img_array[..., 1] >> 2) << 5) | \
                 (img_array[..., 2] >> 3)
        # Ensure that the array is in 16-bit format (each pixel is 2 bytes)
        return rgb565.astype(np.uint16).tobytes(), img.size  # Return size as (width, height)

# Send request to API
def send_request(image_path, convert_to_rgb565=False):
    data = {}
    if convert_to_rgb565:
        rgb565_bytes, (width, height) = convert_image_to_rgb565(image_path)
        files = {
            "file": ("image.rgb565", rgb565_bytes, "application/octet-stream"),
        }
        # Pass width and height in the URL as query parameters
        url_with_params = f"{url2}?width={width}&height={height}"
    else:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            files = {
                "file": ("image.jpg", image_data, "image/jpeg"),
                "format": (None, "RGB888")
            }
        url_with_params = f"{url}?format=RGB888"  # Assuming you want to pass the format in the URL as well

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(url_with_params, files=files)  # Use the full URL with query params
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

send_request(image_path, convert_to_rgb565=True)
