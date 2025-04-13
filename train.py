import os
import cv2
import numpy as np
import face_recognition
from sklearn.preprocessing import LabelEncoder
import pickle
import random
# Directory containing the images of known faces
known_faces_dir = "sample_person"

# List to store the face encodings and labels
known_faces = []
labels = []

def augment_image(image):
    """
    Augments the input image by creating flipped, rotated, brightness, zoom, and noise variations.
    Returns a list of augmented images.
    """
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Horizontal flip
    flipped = cv2.flip(image, 1)  # Horizontal flip
    augmented_images.append(flipped)

    # 90-degree clockwise rotation
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated_90)

    # 90-degree counterclockwise rotation
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(rotated_270)

    # Adjust brightness (random scale between 0.5 and 1.5)
    brightness_scale = random.uniform(0.5, 1.5)
    brightness_image = np.clip(image * brightness_scale, 0, 255).astype(np.uint8)
    augmented_images.append(brightness_image)

    # Random zoom (crop and resize)
    h, w = image.shape[:2]
    zoom_factor = random.uniform(1.1, 1.5)  # Zoom in by 10% to 50%
    center = (w // 2, h // 2)
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    
    # Crop the image and resize
    crop_img = image[center[1] - new_h // 2:center[1] + new_h // 2, center[0] - new_w // 2:center[0] + new_w // 2]
    zoom_img = cv2.resize(crop_img, (w, h))
    augmented_images.append(zoom_img)

    # Add random noise to the image
    noise = np.random.normal(0, 25, image.shape)  # Gaussian noise
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy_image)
    # Add random noise to the image
    noise = np.random.normal(0, 45, image.shape)  # Gaussian noise
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy_image)
    # Add random noise to the image
    noise = np.random.normal(0, 75, image.shape)  # Gaussian noise
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy_image)

    # Adjust contrast (random factor between 0.7 and 1.3)
    contrast_factor = random.uniform(0.7, 1.3)
    contrast_image = np.clip((image - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
    augmented_images.append(contrast_image)

    # Small random rotation
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_small = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated_small)
    # Random occlusion (black rectangle) with random size and position within the image's top-left quadrant. The rectangle is filled with black color. The size of the rectangle is between 20 and 50 pixels wide and 10 and 30 pixels tall. The position is randomly selected within the top-left quadrant of the image. The rectangle is then added to the image. The resulting image is added to the augmented_images list. This augmentation can help to simulate occlusions in the image.
    occluded = image.copy()
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = x1 + random.randint(20, 50)
    y2 = y1 + random.randint(10, 30)
    cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), -1)
    augmented_images.append(occluded)

    # Random HSV jittering (hue, saturation, and value) with random factors. The hue is randomly jittered by adding a random integer between -10 and 10, and the saturation is randomly jittered by multiplying the original saturation by a random factor between 0.8 and 1.2. The resulting image is added to the augmented_images list. This augmentation can help to simulate variations in color and lighting in the image.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-10, 10), 0, 179)  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)  # Saturation
    jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(jittered)

    # Convert the image to grayscale and add it to the augmented_images list. This augmentation can help to simulate images that are not colorful.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    augmented_images.append(gray_bgr)



    return augmented_images
# Iterate through the images in the directory
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            # Apply data augmentation
            augmented_images = augment_image(image)

            for aug_img in augmented_images:
                # Extract face encodings from augmented images
                face_encodings = face_recognition.face_encodings(aug_img)
                
                if len(face_encodings) > 0:
                    known_faces.append(face_encodings[0])
                    labels.append(filename.split(".")[0])
                    print(f"Processed {filename} (augmented)")
            
# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Save the model and label encoder to a file
with open('face_recognition_model.pkl', 'wb') as model_file:
    pickle.dump((known_faces, labels), model_file)
    print("Model saved to face_recognition_model.pkl")

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)
    print("Label encoder saved to label_encoder.pkl")
