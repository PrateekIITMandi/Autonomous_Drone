import onnxruntime as ort
import numpy as np
import cv2
import os
import csv
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8l-pose.onnx")

def extract_number(filename):
    """Extract the numeric part from filenames like frame_1234567890123.png."""
    return filename.split("_")[1].split(".")[0]

def read_depth_from_txt(file_path):
    """Read the depth value from a text file."""
    depth = None
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Average Depth"):
                # Extract numeric value before 'mm'
                parts = line.split(":")
                if len(parts) > 1:
                    depth = parts[1].strip().split()[0]
    return float(depth) if depth else None

def main():
    # Define folders
    pictures_folder = r"C:\Users\prate\Desktop\testing\2.5m\pictures"
    txt_folder = r"C:\Users\prate\Desktop\testing\2.5m\txt"
    output_csv = os.path.join(os.path.dirname(pictures_folder), "results.csv")

    # Collect all image files
    image_files = [f for f in os.listdir(pictures_folder) if f.lower().endswith(".png")]
    image_files.sort()

    # Prepare CSV header
    header = ["Frame_Number", "Width_Pixels", "Height_Pixels", "Area_Pixels", "Depth_mm"]

    # Open CSV for writing
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for img_name in image_files:
            frame_id = extract_number(img_name)
            img_path = os.path.join(pictures_folder, img_name)
            txt_path = os.path.join(txt_folder, f"position_{frame_id}.txt")

            if not os.path.exists(txt_path):
                print(f"⚠️ No matching text file for {img_name}. Skipping.")
                continue

            # Read the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Could not read {img_name}. Skipping.")
                continue

            # Run YOLO inference
            results = model(image)

            # Get detections
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

            # Read depth from text file
            depth = read_depth_from_txt(txt_path)

            # Process each human detection
            for detection in detections:
                if len(detection) < 6:
                    continue
                x1, y1, x2, y2, conf, cls = detection
                if int(cls) != 0:  # only class 0 (person)
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                width = x2 - x1
                height = y2 - y1
                area = width * height

                writer.writerow([frame_id, width, height, area, depth])

    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
