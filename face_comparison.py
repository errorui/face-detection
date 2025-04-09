import os
import shutil
import uuid
import cv2
import numpy as np
import face_recognition
import requests
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Save uploaded files from client
def save_uploaded_files(files):
    saved_paths = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        filename = f"{uuid.uuid4()}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append((filename, path))
    return saved_paths


# Download and save images from URLs
def save_images_from_urls(urls):
    saved_paths = []
    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
                ext = os.path.splitext(url)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    ext = ".jpg"  # Default extension
                filename = f"{uuid.uuid4()}{ext}"
                path = os.path.join(UPLOAD_DIR, filename)
                with open(path, "wb") as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                saved_paths.append((filename, path))
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return saved_paths


# Load and convert images
def load_images(file_paths):
    images = []
    for filename, path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((filename, rgb_img))
    return images


# Face detection + comparison
def detect_and_compare_faces(images):
    face_encodings = {}
    results = []

    for name, img in images:
        face_locations = face_recognition.face_locations(img)
        if not face_locations:
            continue
        encodings = face_recognition.face_encodings(img, face_locations)
        if encodings:
            face_encodings[name] = encodings[0]

    image_names = list(face_encodings.keys())
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            name1 = image_names[i]
            name2 = image_names[j]
            dist = face_recognition.face_distance([face_encodings[name1]], face_encodings[name2])[0]
            match = face_recognition.compare_faces([face_encodings[name1]], face_encodings[name2], tolerance=0.6)[0]

            results.append({
                "image1": name1,
                "image2": name2,
                "match": bool(match),
                "distance": float(dist),
                "similarity": round(float(1.0 - dist), 2)
            })

    return results


@app.get("/")
async def root():
    return {"message": "Welcome to the Face Comparison API!"}


# Compare from uploaded files
@app.post("/compare-faces")
async def compare_faces(files: list[UploadFile] = File(...)):
    saved_files = save_uploaded_files(files)
    if len(saved_files) < 2:
        return JSONResponse(status_code=400, content={"error": "At least 2 valid images are required."})

    images = load_images(saved_files)
    results = detect_and_compare_faces(images)

    for _, path in saved_files:
        os.remove(path)

    if not results:
        return {"message": "No valid faces found in uploaded images."}

    any_match = any(r["match"] for r in results)
    return {
        "result": "YES" if any_match else "NO",
        "details": results
    }


# Compare from image URLs
@app.post("/compare-faces-from-urls")
async def compare_faces_from_urls(request: Request):
    try:
        body = await request.json()
        urls = body.get("urls", [])
        if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
            return JSONResponse(status_code=400, content={"error": "Invalid 'urls' format. Must be a list of strings."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON", "details": str(e)})

    saved_files = save_images_from_urls(urls)
    if len(saved_files) < 2:
        return JSONResponse(status_code=400, content={"error": "At least 2 valid images were not fetched from URLs."})

    images = load_images(saved_files)
    results = detect_and_compare_faces(images)

    for _, path in saved_files:
        os.remove(path)

    if not results:
        return {"message": "No valid faces found in provided image URLs."}

    any_match = any(r["match"] for r in results)
    return {
        "result": "YES" if any_match else "NO",
        "details": results
    }
