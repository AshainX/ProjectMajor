import os
import cv2
import csv
import json
import warnings
import numpy as np
from PIL import Image

# Suppress OpenCV warnings
cv2.setLogLevel(0)
warnings.filterwarnings("ignore")

# Paths to GeoQA dataset
geoqa_json_path = r"C:\Users\ashut\Downloads\GeoQA\GeoQA3\json"  # Folder containing JSON files
geoqa_image_path = r"C:\Users\ashut\Downloads\GeoQA\GeoQA3\image"  # Folder containing images
vocab_file = r"vocabulary_map.json"  # Manually created vocabulary file

# Load vocabulary file
translation_dict = {}
if os.path.exists(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        translation_dict = json.load(f)  # Load mapping
else:
    print("⚠️ Vocabulary file not found! Proceeding without translation.")

# Function to translate Chinese text to English using the vocabulary file
def translate_text(text):
    for cn_word, en_word in translation_dict.items():
        text = text.replace(cn_word, en_word)  # Replace Chinese words with English
    return text

# Shape mapping from Mandarin to English
shape_mapping = {
    "相似三角形": "triangle",
    "勾股定理": "triangle",
    "矩形": "rectangle",
    "正方形": "square",
    "圆周角": "circle"
}

# Function to extract shape name from "formal_point"
def extract_shape(json_data):
    formal_points = json_data.get("formal_point", [])  # Get list from "formal_point"
    
    for point in formal_points:
        translated_point = translate_text(point)  # Convert to English
        
        if translated_point in shape_mapping:
            return shape_mapping[translated_point]  # Return mapped shape name
    
    return "unknown"  # If no shape name is found

# Function to fix PNG files before reading
def fix_png(image_path):
    try:
        img = Image.open(image_path)
        img.save(image_path)  # Overwrite with fixed version
    except Exception as e:
        print(f"❌ Failed to fix image: {image_path}, Error: {e}")

# Process GeoQA dataset (Scan entire directory)
processed_images = []

for root, _, files in os.walk(geoqa_json_path):  # Recursively go through all JSON files
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            image_file = file.replace(".json", ".png")  # Corresponding image file
            image_path = os.path.join(geoqa_image_path, image_file)

            # Read JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

                # Extract shape name from "formal_point"
                shape_label = extract_shape(json_data)

            # Fix the PNG before reading
            fix_png(image_path)

            # Try reading the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                print(f"✅ Processed: {image_path} | Shape: {shape_label}")
                processed_images.append(["GeoQA", file, image_file, shape_label, image_path])
            else:
                print(f"⚠️ OpenCV could not read image even after fixing: {image_path}")

# Save processed data to CSV
output_csv = "GeoQA_Processed.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "json_file", "image_file", "shape", "image_path"])  # CSV header
    writer.writerows(processed_images)

print(f"✅ All images processed and saved to {output_csv}")
