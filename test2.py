import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Load your image (update path to your image)
image_path = r'C:\Users\Raj Raman\Desktop\work\facedetection\sample_person\WhatsApp Image 2025-04-11 at 7.22.05 PM.jpeg'
image = cv2.imread(image_path)

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

    # Random occlusion
    occluded = image.copy()
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = x1 + random.randint(20, 50)
    y2 = y1 + random.randint(10, 30)
    cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), -1)
    augmented_images.append(occluded)

    # Random HSV jittering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-10, 10), 0, 179)  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)  # Saturation
    jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(jittered)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    augmented_images.append(gray_bgr)

    return augmented_images

# Generate augmented images
augmented_images = augment_image(image)

# Display all the results
plt.figure(figsize=(20, 20))
for i, img in enumerate(augmented_images):
    plt.subplot(4, 4, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Augmentation {i}")
    plt.axis('off')

plt.tight_layout()
plt.show()
