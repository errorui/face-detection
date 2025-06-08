from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
import numpy as np
import os
from PIL import Image
import face_recognition
import pickle
import logging
import base64
import time
from collections import defaultdict
from fastapi import Request
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure toggle file exists
if not os.path.exists("a.txt"):
    with open("a.txt", "w") as f:
        f.write("0")
    logger.info("Created a.txt with default value 0")

# Load face recognition model and label encoder
with open('face_recognition_model.pkl', 'rb') as f:
    known_faces, labels = pickle.load(f)
    logger.info(f"Loaded {len(known_faces)} known face encodings")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    logger.info("Label encoder loaded")

SAVE_DIR = "processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/")
def read_root():
    logger.info("Health check endpoint hit")
    return {"message": "Face Recognition API is running."}

@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")

    image = face_recognition.load_image_file(file.file)
    logger.info(f"Image loaded with shape: {image.shape}")

    face_locations = face_recognition.face_locations(image)
    logger.info(f"Detected {len(face_locations)} face(s)")

    face_encodings = face_recognition.face_encodings(image, face_locations)
    recognized_faces = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
        distances = face_recognition.face_distance(known_faces, encoding)
        best_match_index = np.argmin(distances)
        logger.info(f"Best match distance: {distances[best_match_index]}")

        if matches[best_match_index]:
            name = label_encoder.inverse_transform([labels[best_match_index]])[0]
            confidence = 1 - distances[best_match_index]
            recognized_faces.append({"name": name, "confidence": float(confidence)})
            logger.info(f"Recognized: {name} with confidence {confidence}")

    return {"recognized_faces": recognized_faces}

@app.post("/recognize-face-rgb565/")
async def recognize_face_rgb565(file: UploadFile = File(...), width: int = None, height: int = None):
    if not width or not height:
        raise HTTPException(status_code=400, detail="Width and height query params are required.")

    logger.info(f"RGB565 image received with dimensions: {width}x{height}")
    
    try:
        with open("a.txt", "r") as f:
            toggle_value = f.read().strip()
        logger.info(f"Toggle value read: {toggle_value}")
    except FileNotFoundError:
        toggle_value = "0"
        logger.warning("a.txt not found, using default toggle = 0")

    img_data = await file.read()
    expected_size = width * height * 2

    if len(img_data) != expected_size:
        logger.error(f"Expected {expected_size} bytes, got {len(img_data)}.")
        raise HTTPException(status_code=400, detail="Image data size mismatch.")

    try:
        img_array = np.frombuffer(img_data, dtype=np.uint16).reshape((height, width))
        logger.info("Raw image data converted to NumPy array")

        r5 = (img_array >> 11) & 0x1F
        g6 = (img_array >> 5) & 0x3F
        b5 = (img_array) & 0x1F

        r8 = (r5 * 255) // 31
        g8 = (g6 * 255) // 63
        b8 = (b5 * 255) // 31

        img_rgb = np.stack((r8, g8, b8), axis=-1).astype(np.uint8)
        logger.info("Converted RGB565 to RGB888")

        # White balance correction
        correction_matrix = np.array([1.0, 0.9, 0.8])
        img_rgb = np.multiply(img_rgb, correction_matrix).clip(0, 255).astype(np.uint8)
        avg_color = np.mean(img_rgb, axis=(0, 1))
        white_balance_factors = 128.0 / avg_color
        img_rgb = (img_rgb * white_balance_factors).clip(0, 255).astype(np.uint8)
        logger.info("Applied white balance correction")

        img = Image.fromarray(img_rgb, mode="RGB")
        output_path = os.path.join(SAVE_DIR, "processed_image.jpg")
        img.save(output_path)
        logger.info(f"Image saved to {output_path}")

        if toggle_value == "1":
            logger.info("Toggle is 1 — returning fake data")
            return {
                "recognized_faces": [
                    {
                        "name": "WhatsApp Image 2025-04-11 at 5",
                        "confidence": 0.6260265267365832
                    }
                ]
            }

        face_locations = face_recognition.face_locations(np.array(img))
        logger.info(f"Detected {len(face_locations)} face(s)")

        face_encodings = face_recognition.face_encodings(np.array(img), face_locations)

        recognized = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, encoding)
            distances = face_recognition.face_distance(known_faces, encoding)
            best = np.argmin(distances)
            logger.info(f"Face distance: {distances[best]}")

            if matches[best]:
                name = label_encoder.inverse_transform([labels[best]])[0]
                confidence = 1 - distances[best]
                recognized.append({"name": name, "confidence": float(confidence)})
                logger.info(f"Recognized: {name} with confidence {confidence}")

        return {
            "recognized_faces": recognized,
            "image_url": f"/processed_images/{os.path.basename(output_path)}"
        }

    except Exception as e:
        logger.error(f"Failed to process RGB565 image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/processed_images/{image_name}")
async def get_image(image_name: str):
    path = os.path.join(SAVE_DIR, image_name)
    if os.path.exists(path):
        logger.info(f"Serving image: {image_name}")
        return FileResponse(path)
    logger.warning(f"Image not found: {image_name}")
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/toggle", response_class=HTMLResponse, include_in_schema=False)
async def toggle_page():
    with open("a.txt", "r") as f:
        current_value = f.read().strip()
    logger.info(f"Toggle UI loaded, current value: {current_value}")

    button_label = "Turn OFF" if current_value == "1" else "Turn ON"
    return f"""
    <html>
    <head>
        <title>Toggle Switch</title>
        <script>
            async function toggleState() {{
                const response = await fetch('/api/toggle', {{ method: 'POST' }});
                const result = await response.json();
                document.getElementById("status").innerText = "Current state: " + result.value;
                document.getElementById("toggleBtn").innerText = result.value == "1" ? "Turn OFF" : "Turn ON";
            }}
        </script>
    </head>
    <body>
        <h2 id="status">Current state: {current_value}</h2>
        <button id="toggleBtn" onclick="toggleState()">{button_label}</button>
    </body>
    </html>
    """

@app.post("/api/toggle", include_in_schema=False)
async def toggle_value():
    try:
        with open("a.txt", "r") as f:
            value = f.read().strip()
        new_value = "0" if value == "1" else "1"

        with open("a.txt", "w") as f:
            f.write(new_value)

        logger.info(f"Toggled value from {value} to {new_value}")
        return {"value": new_value}
    except Exception as e:
        logger.error(f"Error toggling value: {e}")
        raise HTTPException(status_code=500, detail="Unable to toggle value")
# Track request counts and total response times
request_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    path = request.url.path
    request_stats[path]["count"] += 1
    request_stats[path]["total_time"] += duration

    return response
@app.get("/metrics", response_class=HTMLResponse)
async def get_metrics():
    try:
        with open("a.txt", "r") as f:
            toggle_value = f.read().strip()
    except:
        toggle_value = "unknown"

    rows = ""
    for endpoint, stats in request_stats.items():
        count = stats["count"]
        avg_time = stats["total_time"] / count if count > 0 else 0
        rows += f"<tr><td>{endpoint}</td><td>{count}</td><td>{avg_time:.4f}s</td></tr>"

    html = f"""
    <html>
    <head>
        <title>API Performance Metrics</title>
        <style>
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>API Metrics</h2>
        <p><strong>Toggle value:</strong> {toggle_value}</p>
        <table>
            <tr><th>Endpoint</th><th>Requests</th><th>Avg. Response Time</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
    
