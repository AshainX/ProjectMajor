import os
import cv2
import csv
import warnings
import json
import numpy as np
from PIL import Image

# Suppress OpenCV warnings
cv2.setLogLevel(0)  # Works on all OpenCV versions
warnings.filterwarnings("ignore")

# Manually specify dataset paths
dataset_paths = {
    "aaai": r"C:\Users\ashut\Downloads\GeoS\aaai",
    "official": r"C:\Users\ashut\Downloads\GeoS\officiall",
    "practice": r"C:\Users\ashut\Downloads\GeoS\practice"
}

# Shape keywords mapping
shape_keywords = {
    "circle": ["circle", "radius", "diameter"],
    "triangle": ["triangle", "hypotenuse", "isosceles", "scalene", "equilateral"],
    "square": ["square"],
    "rectangle": ["rectangle"],
    "trapezium": ["trapezium", "trapezoid"],
    "parallelogram": ["parallelogram"],
    "pentagon": ["pentagon"],
    "hexagon": ["hexagon"]
}

# CSV output file
output_csv = "processed_images.csv"

# Store processed images with shape names
processed_images = []

# Function to extract shape name from text
def extract_shape(text):
    for shape, keywords in shape_keywords.items():
        if any(word in text.lower() for word in keywords):
            return shape
    return "unknown"  # If no shape name is found

# Function to fix PNG files before reading
def fix_png(image_path):
    try:
        img = Image.open(image_path)
        img.save(image_path)  # Overwrite with fixed version
    except Exception as e:
        print(f"❌ Failed to fix image: {image_path}, Error: {e}")

# Process each dataset folder
for dataset_name, dataset_path in dataset_paths.items():
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Folder '{dataset_path}' not found! Skipping...")
        continue

    # Iterate through JSON files (to extract shape names)
    for file in os.listdir(dataset_path):
        if file.endswith(".json"):  
            json_path = os.path.join(dataset_path, file)
            image_file = file.replace(".json", ".png")  # Corresponding image file
            image_path = os.path.join(dataset_path, image_file)

            # Read JSON file
            with open(json_path, "r") as f:
                json_data = json.load(f)
                question_text = json_data.get("text", "")

                # Extract shape name
                shape_label = extract_shape(question_text)

            # Fix the PNG before reading
            fix_png(image_path)

            # Try reading the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                print(f"✅ Processed: {image_path} | Shape: {shape_label}")
                processed_images.append([dataset_name, file, image_file, shape_label, image_path])
            else:
                print(f"⚠️ OpenCV could not read image even after fixing: {image_path}")

# Save processed images to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "json_file", "image_file", "shape", "image_path"])  # CSV header
    writer.writerows(processed_images)

print(f"All images processed and saved to {output_csv}")
