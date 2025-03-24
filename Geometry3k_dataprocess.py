import os
import json
import csv

# Base path to Geometry3K dataset
geometry3k_path = r"C:\Users\ashut\Downloads\Geometry3K"  # Update this path

# Folders to process
dataset_types = ["train", "test", "val"]

# Output CSV file
output_csv = "Geometry3K_Processed.csv"

# Define valid shape names
valid_shapes = {
    "triangle", "rectangle", "parallelogram", "trapezium", "circle", "square"
}

# Function to extract shape name from logic_form.json
def extract_shape(logic_data):
    """Extracts the shape name from multiple fields in logic_form.json."""
    # Check in text logic fields
    logic_fields = logic_data.get("text_logic_form", []) + logic_data.get("dissolved_text_logic_form", [])
    
    for logic in logic_fields:
        if "(" in logic and ")" in logic:
            shape = logic.split("(")[0].strip().lower()  # Extract shape name before '('
            if shape in valid_shapes:
                return shape  # Return valid shape name

    return "unknown"  # If no valid shape is found

# Process dataset folders
processed_data = []

for dataset in dataset_types:
    dataset_path = os.path.join(geometry3k_path, dataset)

    # Loop through each subfolder
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue  # Skip files, process only directories

        # Define file paths
        logic_form_path = os.path.join(subfolder_path, "logic_form.json")
        image_path_1 = os.path.join(subfolder_path, "img_diagram_point.png")  # Preferred image
        image_path_2 = os.path.join(subfolder_path, "img_diagram.png")  # Alternative

        # Check if logic_form.json exists
        if not os.path.exists(logic_form_path):
            print(f"⚠️ Missing logic_form.json in: {subfolder_path}")
            continue

        # Read logic_form.json
        with open(logic_form_path, "r", encoding="utf-8") as f:
            logic_data = json.load(f)
            shape_label = extract_shape(logic_data)

        # Use the preferred image, fallback if not found
        image_folder = subfolder if os.path.exists(image_path_1) or os.path.exists(image_path_2) else "No Image Found"

        # Store dataset info
        print(f"✅ Processed: {subfolder} | Shape: {shape_label}")
        processed_data.append([dataset, subfolder, shape_label, image_folder, subfolder_path])  # Added location column

# Save processed data to CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "folder_name", "shape", "image_folder", "location"])  # Updated header
    writer.writerows(processed_data)

print(f"✅ All data processed and saved to {output_csv}")
