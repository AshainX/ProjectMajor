# import os
# import cv2
# import csv
# import math
# import numpy as np
# import pandas as pd

# # === Step 1: Load CSV Files ===
# csv_files = {
#     "GeoS": "GeoS_Processed.csv",
#     "GeoQA": "GeoQA_Processed.csv",
#     "Geometry3K": "Geometry3K_Processed.csv"
# }

# # Load datasets into Pandas DataFrames
# df_dict = {}
# for dataset, file in csv_files.items():
#     if os.path.exists(file):
#         try:
#             df_dict[dataset] = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", dtype={"folder_name": str})
#         except Exception as e:
#             print(f"❌ Error loading {file}: {e}")
#     else:
#         print(f"⚠️ CSV file not found: {file}")

# # === Step 2: Shape Detection Using OpenCV (Algorithm 1-7) ===
# def detect_shape(image_path):
#     """Detects the shape in the given image using contour-based shape detection."""
#     img = cv2.imread(image_path)

#     if img is None:
#         print(f"❌ Image not found: {image_path}")
#         return "unknown"

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Gaussian blur to reduce noise (Algorithm 3)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
#     # Edge detection using Canny (Algorithm 4)
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         return "unknown"

#     # Select the largest contour
#     contour = max(contours, key=cv2.contourArea)

#     # Approximate contour to polygon (Algorithm 5)
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     # Count vertices
#     vertices_count = len(approx)

#     # Compute area & perimeter
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)

#     # === Shape Classification (Algorithms 6-7) ===
#     shape = "unknown"

#     if vertices_count == 3:
#         shape = classify_triangle(approx)
#     elif vertices_count == 4:
#         shape = classify_quadrilateral(approx)
#     elif vertices_count > 4:
#         circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0
#         if circularity > 0.5:
#             shape = "circle"
#         else:
#             shape = "unknown"  # Avoid polygon classification

#     return shape

# # === Step 3: Triangle Classification (Algorithm 6) ===
# def classify_triangle(approx):
#     """Classifies triangles as equilateral, isosceles, or scalene."""
#     side_lengths = [math.dist(approx[i][0], approx[(i + 1) % 3][0]) for i in range(3)]
    
#     a, b, c = side_lengths
#     angles = sorted([
#         math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c))),
#         math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c))),
#         math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b))),
#     ])

#     if a == b == c:
#         return "equilateral triangle"
#     elif a == b or b == c or a == c:
#         if 89 < angles[2] < 91:
#             return "isosceles right triangle"
#         elif angles[2] > 90:
#             return "isosceles obtuse triangle"
#         return "isosceles acute triangle"
#     else:
#         if 89 < angles[2] < 91:
#             return "scalene right triangle"
#         elif angles[2] > 90:
#             return "scalene obtuse triangle"
#         return "scalene acute triangle"

# # === Step 4: Quadrilateral Classification (Algorithm 7) ===
# def classify_quadrilateral(approx):
#     """Classifies quadrilaterals based on side lengths and angles."""
#     side_lengths = [math.dist(approx[i][0], approx[(i + 1) % 4][0]) for i in range(4)]
#     angles = []

#     for i in range(4):
#         p0, p1, p2 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
#         v1 = (p0[0] - p1[0], p0[1] - p1[1])
#         v2 = (p2[0] - p1[0], p2[1] - p1[1])
#         dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
#         mag1 = math.dist(p0, p1)
#         mag2 = math.dist(p1, p2)
#         angle = math.degrees(math.acos(dot_prod / (mag1 * mag2))) if mag1 and mag2 else 0
#         angles.append(angle)

#     is_square = all(abs(side_lengths[i] - side_lengths[0]) < 5 for i in range(1, 4))
#     is_rectangle = all(abs(a - 90) < 10 for a in angles)

#     if is_square:
#         return "square"
#     elif is_rectangle:
#         return "rectangle"
#     elif abs(side_lengths[0] - side_lengths[2]) < 5 and abs(side_lengths[1] - side_lengths[3]) < 5:
#         return "parallelogram"
#     elif abs(side_lengths[0] - side_lengths[1]) < 5 or abs(side_lengths[2] - side_lengths[3]) < 5:
#         return "kite"
#     elif angles.count(90) == 1:
#         return "trapezium"
    
#     return "quadrilateral"

# # === Step 5: Get Image Paths from CSV ===
# def get_image_path(row, dataset):
#     """Finds the correct image path based on dataset structure."""
#     if dataset == "Geometry3K":
#         base_path = os.path.dirname(row["image_path"])
#         image_folder = str(row["folder_name"])

#         img_path_1 = os.path.join(base_path, image_folder, "img_diagram.png")
#         img_path_2 = os.path.join(base_path, image_folder, "img_diagram_point.png")

#         return img_path_1 if os.path.exists(img_path_1) else img_path_2 if os.path.exists(img_path_2) else None

#     return row["image_path"]

# # === Step 6: Process & Save Results to CSV ===
# output_csv = "Shape_Predictions_v2.csv"
# results = []

# def evaluate_images():
#     for dataset, df in df_dict.items():
#         for _, row in df.iterrows():
#             if row["shape"].lower() == "unknown":
#                 continue

#             image_path = get_image_path(row, dataset)

#             if image_path is None or not os.path.exists(image_path):
#                 continue  

#             detected_shape = detect_shape(image_path)
#             if detected_shape is None:
#                 continue

#             is_correct = "Yes" if detected_shape == row["shape"].lower() else "No"

#             results.append([dataset, row["image_path"], row["shape"], detected_shape, is_correct])

#     with open(output_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["dataset", "image_path", "expected_shape", "shape_predicted", "is_shape_predicted"])
#         writer.writerows(results)

#     print("✅ Shape Predictions Saved!")

# evaluate_images()


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
        try:
            df_dict[dataset] = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", dtype={"folder_name": str})
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    else:
        print(f"⚠️ CSV file not found: {file}")

# === Step 2: Shape Detection Using OpenCV ===
def detect_shape(image_path):
    """Detects the shape in the given image using contour-based shape detection."""
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Image not found: {image_path}")
        return "unknown"

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Edge detection using Canny
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

    # Compute area & perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # === Shape Classification ===
    shape = "unknown"

    if vertices_count == 3:
        return "triangle"  # Always classify as triangle

    elif vertices_count == 4:
        return classify_quadrilateral(approx)  # Must be classified as Square, Rectangle, Trapezium, or Parallelogram

    else:
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        if circularity > 0.5:
            return "circle"  # Circle detection remains independent

    return shape

# === Step 3: Quadrilateral Classification ===
def classify_quadrilateral(approx):
    """Classifies quadrilaterals as Square, Rectangle, Trapezium, or Parallelogram.
       If none of these match, returns 'unknown'. """

    side_lengths = [math.dist(approx[i][0], approx[(i + 1) % 4][0]) for i in range(4)]
    angles = []

    for i in range(4):
        p0, p1, p2 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.dist(p0, p1)
        mag2 = math.dist(p1, p2)
        angle = math.degrees(math.acos(dot_prod / (mag1 * mag2))) if mag1 and mag2 else 0
        angles.append(angle)

    is_square = all(abs(side_lengths[i] - side_lengths[0]) < 5 for i in range(1, 4))
    is_rectangle = all(abs(a - 90) < 10 for a in angles)

    if is_square:
        return "square"
    elif is_rectangle:
        return "rectangle"
    elif abs(side_lengths[0] - side_lengths[2]) < 5 and abs(side_lengths[1] - side_lengths[3]) < 5:
        return "parallelogram"
    elif angles.count(90) == 1:
        return "trapezium"

    return "unknown"  # No "quadrilateral" output, only specific shapes or "unknown"

# === Step 4: Get Image Paths from CSV ===
def get_image_path(row, dataset):
    """Finds the correct image path based on dataset structure."""
    if dataset == "Geometry3K":
        base_path = os.path.dirname(row["image_path"])
        image_folder = str(row["folder_name"])

        img_path_1 = os.path.join(base_path, image_folder, "img_diagram.png")
        img_path_2 = os.path.join(base_path, image_folder, "img_diagram_point.png")

        return img_path_1 if os.path.exists(img_path_1) else img_path_2 if os.path.exists(img_path_2) else None

    return row["image_path"]

# === Step 5: Process & Save Results to CSV ===
output_csv = "Shape_PredictionsV4.csv"
results = []

def evaluate_images():
    """Processes all images in CSV, predicts shapes, and saves results."""
    for dataset, df in df_dict.items():
        for _, row in df.iterrows():
            if row["shape"].lower() not in ["triangle", "square", "rectangle", "trapezium", "parallelogram", "circle"]:
                continue  # Skip other shapes

            image_path = get_image_path(row, dataset)

            if image_path is None or not os.path.exists(image_path):
                continue  

            detected_shape = detect_shape(image_path)
            if detected_shape is None:
                continue

            is_correct = "Yes" if detected_shape == row["shape"].lower() else "No"

            results.append([dataset, row["image_path"], row["shape"], detected_shape, is_correct])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image_path", "expected_shape", "shape_predicted", "is_shape_predicted"])
        writer.writerows(results)

    print("✅ Shape Predictions Saved!")

evaluate_images()

