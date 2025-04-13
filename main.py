from fastapi import FastAPI, File, UploadFile
import face_recognition
import pickle
import numpy as np


app = FastAPI()

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
