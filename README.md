# Computer Vision: RGB–Thermal Alignment & Change Detection

This repository contains two computer vision tasks focused on real-world image processing challenges:

1. **RGB–Thermal Image Alignment (Feature-Based Homography)**
2. **Scene Change Detection using Image Differencing**

Both tasks were implemented using Python and OpenCV with fully automated batch processing pipelines.

---

# Project Overview:

## Task 1- RGB–Thermal Image Alignment

### Objective:
Align thermal images with their corresponding RGB images and generate corrected thermal outputs while keeping the RGB image unchanged.

### Input Naming Format:
XXXX_T.JPG → Thermal image
XXXX_Z.JPG → RGB image


Images were captured from **two different cameras**, so they are not perfectly aligned by default.

---

## Approach:

### 1️. Automatic Pair Matching:
- Extracts shared identifiers from filenames
- Matches `_T.JPG` and `_Z.JPG` pairs

### 2️. Feature Detection & Matching:
- SIFT feature detection (if available)
- FLANN-based matcher
- Lowe’s ratio test for robust keypoint filtering

### 3️. Homography Estimation:
- RANSAC-based homography computation
- Validation checks:
  - Minimum inliers threshold
  - Condition number stability check
  - Non-black pixel coverage validation

### 4️. Perspective Warping:
- Applies homography transformation
- Generates aligned thermal image: `XXXX_AT.JPG`

### 5️. Fallback Mechanism:
If homography fails:
- Safe scaling fallback is applied
- Ensures output generation for all image pairs

---

## Task 1 Output Structure:

Task1/
├── task_1_code.py
└── task_1_output/
├── XXXX_Z.JPG
├── XXXX_AT.JPG
└── diagnostics.json


---

# Task 2- Change Detection Algorithm:

### 🎯 Objective
Detect missing objects between two perfectly aligned images and highlight them in the "after" image.

### Input Naming Format:

X.jpg → Before image
X~2.jpg → After image


Images are guaranteed to be 100% aligned.

---

## Approach:

### 1️. Pair Identification:
Automatically detects before–after pairs.

### 2️. Image Differencing:
- Absolute pixel difference
- Grayscale conversion
- Adaptive / Otsu thresholding

### 3. Noise Reduction:
- Morphological opening & closing
- Area-based filtering to remove small artifacts

### 4️. Region Detection:
- Connected component analysis
- Bounding box extraction

### 5. Annotation:
- Red bounding boxes drawn around missing objects
- Output saved as: `X~3.jpg`

---

## Task 2 Output Structure:


Task2/
├── task_2_code.py
└── task_2_output/
├── X.jpg
└── X~3.jpg


Original `X~2.jpg` images are not included in the output.

---

# 🛠️ Technologies Used:

- Python 3
- OpenCV
- NumPy
- SIFT Feature Detection
- RANSAC Homography
- Morphological Image Processing
- Connected Component Analysis

---

# How To Run:

## Task 1
```bash
python task_1_code.py --input ./input-images --output ./task_1_output
Task 2
python task_2_code.py --input ./input-images --output ./task_2_output
