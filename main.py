from PIL import Image
from PIL import ImageFilter
from fastapi import FastAPI, File, UploadFile
import face_recognition
import pickle
import numpy as np
import os
from fastapi.responses import FileResponse
app = FastAPI()
import logging
from fastapi import HTTPException
# Configure the logger
logging.basicConfig(level=logging.INFO)  # You can change the level to DEBUG if you need more details
logger = logging.getLogger(__name__)



# Load the model and label encoder
with open('face_recognition_model.pkl', 'rb') as model_file:
    known_faces, labels = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.get('/')
def read_root():
    return {"message": "Welcome to the Face Recognition API!"}

@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    # Read the uploaded image
    image = face_recognition.load_image_file(file.file)

    # Find face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Encode the faces in the image
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Initialize a list to store the recognized faces and their confidence scores
    recognized_faces = []

    for face_encoding in face_encodings:
        # Compare the face encoding with the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        # Find the best match
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = label_encoder.inverse_transform([labels[best_match_index]])[0]
            confidence = 1 - face_distances[best_match_index]
            recognized_faces.append({"name": name, "confidence": float(confidence)})

    return {"recognized_facess": recognized_faces}

SAVE_DIR = "processed_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@app.post("/recognize-face-rgb565/")
async def recognize_face_rgb565(file: UploadFile = File(...), width: int = None, height: int = None):
    if width and height:
        logger.info(f"Received RGB565 image with width: {width}, height: {height}, format: RGB565")
    else:
        logger.error("Width or height is missing from the request!")
        return {"error": "Width and height are required for RGB565 images"}

    img_data = await file.read()
    logger.info(f"Received image byte size: {len(img_data)} bytes")

    expected_size = width * height * 2
    if len(img_data) != expected_size:
        logger.error(f"Expected {expected_size} bytes for the image, but got {len(img_data)} bytes.")
        return {"error": "Mismatch between image size and dimensions provided"}

    img_array = np.frombuffer(img_data, dtype=np.uint16)

    try:
        img_array = img_array.reshape((height, width))

        # Convert RGB565 to RGB888 with proper scaling
        red = ((img_array >> 11) & 0x1F) << 3
        green = ((img_array >> 5) & 0x3F) << 2
        blue = (img_array & 0x1F) << 3
        
        # Stack channels to form the RGB image
        img_array_rgb888 = np.stack((red, green, blue), axis=-1).astype(np.uint8)
        
        # Convert to a PIL Image for further processing
        img = Image.fromarray(img_array_rgb888, mode='RGB')
        
        # Optionally apply a blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Save the processed image
        
        output_filename = os.path.join(SAVE_DIR, "processed_image.jpg")
        output_filename = output_filename.replace("\\", "/")  # Convert to Unix-style path
        img.save(output_filename, format="JPEG")
        logger.info(f"Saved processed image to {output_filename}")

        # Run face recognition
        face_locations = face_recognition.face_locations(img_array_rgb888)
        face_encodings = face_recognition.face_encodings(img_array_rgb888, face_locations)

        recognized_faces = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = label_encoder.inverse_transform([labels[best_match_index]])[0]
                confidence = 1 - face_distances[best_match_index]
                recognized_faces.append({"name": name, "confidence": float(confidence)})

        # Generate a URL for the processed image
        image_url = f"/processed_images/{os.path.basename(output_filename)}"
        return {"recognized_faces": recognized_faces, "image_url": image_url}

    except Exception as e:
        logger.error(f"Error processing RGB565 image: {e}")
        return {"error": "Failed to process the image"}

# Serve the processed image from a public URL
@app.get("/processed_images/{image_name}")
async def serve_processed_image(image_name: str):
    image_path = os.path.join(SAVE_DIR, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")