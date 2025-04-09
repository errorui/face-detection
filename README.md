# Face Detection and Comparison System

This project provides a simple face detection and comparison system to determine if the same person appears in different photos.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: The face_recognition library requires dlib, which may need additional setup:
- On Windows, you might need Visual C++ build tools
- On Linux, you might need to install cmake and other build dependencies

### 2. Prepare Sample Images

Place photos in the `sample_person` directory. The system works best with:
- Clear, well-lit photos
- Faces that are clearly visible (not obscured or at extreme angles)
- One face per image (for best results)

### 3. Run the Face Comparison

```bash
python face_comparison.py
```

### 4. View Results

The script will:
- Print comparison results in the console
- Generate comparison images showing whether faces match
- Save these images in the current directory with names like `comparison_image1_image2.jpg`

## How It Works

1. The system loads all images from the `sample_person` directory
2. For each image, it detects faces and extracts facial features
3. It compares the facial features between each pair of images
4. It determines if the faces belong to the same person based on similarity
5. It generates visual comparisons showing the results

## Interpreting Results

- **Similarity Score**: Higher values (closer to 1.0) indicate more similar faces
- **Distance**: Lower values indicate more similar faces
- **Same Person**: Boolean result based on whether the similarity exceeds a threshold

## Limitations

- Works best with clear, front-facing photos
- May struggle with poor lighting, extreme angles, or partially obscured faces
- Performance depends on the quality of the input images
