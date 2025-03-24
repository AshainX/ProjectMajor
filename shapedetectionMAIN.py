import os
import cv2
import csv
import math
import numpy as np
import pandas as pd

# === Step 1: Load CSV Files ===
csv_files = {
    "GeoS": "GeoS_Processed.csv",
    "GeoQA": "GeoQA_Processed.csv",
    "Geometry3K": "Geometry3K_Processed.csv"
}

# Load datasets into Pandas DataFrames
df_dict = {}
for dataset, file in csv_files.items():
    if os.path.exists(file):
        df_dict[dataset] = pd.read_csv(file)
    else:
        print(f"⚠️ CSV file not found: {file}")

# === Step 2: Preprocessing & Shape Detection ===
def detect_shape(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Image not found: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "unknown"

    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Count vertices
    vertices_count = len(approx)

    # Center
    M = cv2.moments(contour)
    cx, cy = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) if M['m00'] != 0 else (0, 0)

    # Area & Perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # === Shape Classification ===
    shape = "unknown"

    if vertices_count == 3:
        shape = "triangle"
    elif vertices_count == 4:
        # Get angles to distinguish rectangle/square
        side_lengths = [math.hypot(approx[i][0][0] - approx[(i + 1) % 4][0][0], 
                                   approx[i][0][1] - approx[(i + 1) % 4][0][1]) for i in range(4)]

        angles = []
        for i in range(4):
            p0, p1, p2 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.hypot(v1[0], v1[1])
            mag2 = math.hypot(v2[0], v2[1])
            cos_theta = dot_prod / (mag1 * mag2) if mag1 and mag2 else 0
            angle = math.acos(min(max(cos_theta, -1), 1)) * 180 / math.pi
            angles.append(angle)

        is_square = all(abs(side_lengths[i] - side_lengths[0]) < 5 for i in range(1, 4))
        is_rectangle = all(abs(a - 90) < 10 for a in angles)

        if is_square:
            shape = "square"
        elif is_rectangle:
            shape = "rectangle"
        else:
            shape = "quadrilateral"

    elif vertices_count > 4:
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        shape = "circle" if circularity > 0.8 else "polygon"

    return shape

# === Step 3: Compare with CSV Data & Compute Accuracy ===
def find_expected_shape(image_name, folder_name):
    """Searches all CSVs for the expected shape of an image."""
    for dataset, df in df_dict.items():
        if dataset == "Geometry3K":
            match = df[(df["folder_name"] == folder_name)]
        else:
            match = df[(df["image_file"] == image_name)]

        if not match.empty:
            expected_shape = match.iloc[0]["shape"].lower()
            if expected_shape != "unknown":
                return expected_shape
    return None  # If no valid match found

# === Step 4: Evaluate a Batch of Images ===
def evaluate_images(image_folder):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    total_images = 0

    for dataset, df in df_dict.items():
        for _, row in df.iterrows():
            if row["shape"].lower() == "unknown":
                continue  # Skip unknown shapes

            # Find image path based on dataset structure
            if dataset == "Geometry3K":
                image_path = os.path.join(image_folder, dataset, row["folder_name"], "img_diagram.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, dataset, row["folder_name"], "img_diagram_point.png")
            else:
                image_path = os.path.join(image_folder, dataset, row["image_file"])

            if not os.path.exists(image_path):
                continue  # Skip if image is missing

            expected_shape = row["shape"].lower()
            detected_shape = detect_shape(image_path)

            if detected_shape is None:
                continue  # Skip invalid detections

            total_images += 1
            if detected_shape == expected_shape:
                true_positive += 1
            else:
                false_positive += 1

    # Calculate accuracy metrics
    accuracy = (true_positive / total_images) * 100 if total_images > 0 else 0
    print("\n=== Evaluation Results ===")
    print(f"Total Images Evaluated: {total_images}")
    print(f"True Positives: {true_positive}")
    print(f"False Positives: {false_positive}")
    print(f"Accuracy: {accuracy:.2f}%")

# === Step 5: Run the Evaluation ===
image_dataset_folder = r"C:\Users\ashut\Downloads\Geometry3K"  # Update with your dataset path
evaluate_images(image_dataset_folder)
