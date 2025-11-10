"""
Human Pose and 3D Distance Estimation

This script performs real-time human detection, pose estimation, and 3D position
calculation using a YOLOv8-Pose model. It supports both standard webcams and
Intel RealSense depth cameras.

Features:
- Option to use Webcam or Intel RealSense.
- Automatic use of RealSense intrinsic calibration for higher accuracy.
- Person detection filtered by a confidence threshold.
- Calculation of person's orientation (tilt) based on shoulder position.
- AI-based distance calculation corrected by the person's orientation.
- Real-time display of FPS, distance, and orientation on the video feed.
- Detailed terminal output of X, Y, Z coordinates and sensor depth.
- Automatic saving of annotated frames every 5 seconds or on new person detection.
- Comprehensive logging of all measurements and calculated errors to a timestamped CSV file.
"""
from ultralytics import YOLO
import cv2
import time
import numpy as np
import math
import os
import csv
from datetime import datetime

# --- Helper Functions ---

def calculate_com(points):
    """Calculates the Center of Mass from a list of keypoints."""
    valid_points = [p for p in points if p[0] > 0 and p[1] > 0]
    if not valid_points:
        return None
    valid_points_np = np.array(valid_points)
    com = np.mean(valid_points_np, axis=0)
    return tuple(com.astype(int))

def estimate_distance(known_width, focal_length, pixel_width):
    """Estimates Z distance based on a known width, focal length, and pixel width."""
    if pixel_width < 1e-5:
        return 0
    return (known_width * focal_length) / pixel_width

def calculate_orientation(p1, p2):
    """Calculates the angle of the line connecting two points with the horizontal axis."""
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg, angle_rad

# --- Main Function ---

def main():
    # --- Configuration ---
    USE_REALSENSE = True         # Set to True for Intel RealSense, False for webcam
    CONFIDENCE_THRESHOLD = 0.80  # Detection confidence threshold (80%)
    MODEL_PATH = "yolov8x-pose.onnx" # Path to your ONNX model
    SAVE_DIR = "saved_frames"    # Directory to save interesting frames
    MAX_SENSOR_DEPTH = 14.0      # Max reliable distance for the depth sensor in meters
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- Constants for Calculations ---
    AVG_SHOULDER_WIDTH = 0.45 # Average shoulder width in meters

    # --- Load Model ---
    print(f"Loading optimized model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # --- CSV Logging Setup ---
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_filename, 'w', newline='')
    csv_writer = csv.writer(log_file)
    header = [
        "Timestamp", "Person_ID", "AI_Distance_m", "Sensor_Distance_m",
        "Absolute_Error_m", "Absolute_Percentage_Error", "Orientation_deg",
        "Coord_X_m", "Coord_Y_m"
    ]
    csv_writer.writerow(header)
    print(f"Logging data to {log_filename}")

    # --- Initialize Camera and Get Calibration Data ---
    pipeline = None
    if USE_REALSENSE:
        try:
            import pyrealsense2 as rs
            print("Initializing Intel RealSense Camera...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            FOCAL_LENGTH = intrinsics.fx
            FRAME_CENTER_X = intrinsics.ppx
            FRAME_CENTER_Y = intrinsics.ppy
            print(f"Using RealSense Intrinsics: Focal Length={FOCAL_LENGTH:.2f}px, Center=({FRAME_CENTER_X:.2f}, {FRAME_CENTER_Y:.2f})px")
        except ImportError:
            print("Error: pyrealsense2 is not installed. Please install it to use the RealSense camera.")
            USE_REALSENSE = False
        except RuntimeError as e:
            print(f"Error initializing RealSense: {e}. Is the camera connected?")
            USE_REALSENSE = False

    if not USE_REALSENSE:
        print("Initializing Webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Using manual calibration for webcam.")
        KNOWN_DISTANCE = 0.6; KNOWN_WIDTH = 0.3; PIXEL_WIDTH_IN_IMAGE = 200
        FOCAL_LENGTH = (PIXEL_WIDTH_IN_IMAGE * KNOWN_DISTANCE) / KNOWN_WIDTH
        FRAME_CENTER_X = 640 / 2; FRAME_CENTER_Y = 480 / 2
        print(f"Using Manual Calibration: Focal Length={FOCAL_LENGTH:.2f}px")

    # --- Main Loop Variables ---
    prev_time = 0
    last_save_time = time.time()
    previous_person_count = 0

    try:
        while True:
            # --- Get Frame ---
            if USE_REALSENSE:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()
                if not color_frame_rs or not depth_frame_rs: continue
                frame = np.asanyarray(color_frame_rs.get_data())
            else:
                ret, frame = cap.read()
                if not ret: break
            
            annotated_frame = frame.copy()
            
            # --- Inference ---
            results = model(frame, verbose=False)
            current_person_count = 0
            
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    if boxes.conf[i] >= CONFIDENCE_THRESHOLD:
                        current_person_count += 1
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        
                        if result.keypoints and len(result.keypoints.xy) > i:
                            keypoints = result.keypoints.xy.cpu().numpy()[i]
                            l_shoulder, r_shoulder = keypoints[5], keypoints[6]
                            com_global = calculate_com(keypoints)

                            if (l_shoulder[0] > 0 and r_shoulder[0] > 0) and com_global:
                                orientation_deg, orientation_rad = calculate_orientation(l_shoulder, r_shoulder)
                                pixel_shoulder_width = math.sqrt((l_shoulder[0] - r_shoulder[0])**2 + (l_shoulder[1] - r_shoulder[1])**2)
                                corrected_shoulder_width = AVG_SHOULDER_WIDTH * abs(math.cos(orientation_rad))
                                corrected_ai_z = estimate_distance(corrected_shoulder_width, FOCAL_LENGTH, pixel_shoulder_width)
                                
                                meters_per_pixel = AVG_SHOULDER_WIDTH / pixel_shoulder_width if pixel_shoulder_width > 0 else 0
                                calculated_x = (com_global[0] - FRAME_CENTER_X) * meters_per_pixel
                                calculated_y = (com_global[1] - FRAME_CENTER_Y) * meters_per_pixel
                                
                                sensor_depth_m = 0
                                if USE_REALSENSE:
                                    sensor_depth_m = depth_frame_rs.get_distance(com_global[0], com_global[1])
                                    if sensor_depth_m > MAX_SENSOR_DEPTH or sensor_depth_m == 0:
                                        sensor_depth_m = 0

                                    if sensor_depth_m > 0:
                                        abs_error = abs(corrected_ai_z - sensor_depth_m)
                                        abs_percent_error = (abs_error / sensor_depth_m) * 100
                                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                        log_data = [timestamp, current_person_count, f"{corrected_ai_z:.3f}", f"{sensor_depth_m:.3f}", f"{abs_error:.3f}", f"{abs_percent_error:.2f}", f"{orientation_deg:.1f}", f"{calculated_x:.3f}", f"{calculated_y:.3f}"]
                                        csv_writer.writerow(log_data)
                                
                                term_output = f"Person {current_person_count}: X:{calculated_x:.2f}m Y:{calculated_y:.2f}m | AI_Z:{corrected_ai_z:.2f}m | Sensor_Z:{sensor_depth_m:.2f}m | Tilt:{orientation_deg:.1f}deg"
                                print(term_output)

                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                info_text = f"Dist: {corrected_ai_z:.2f}m | Tilt: {orientation_deg:.1f} deg"
                                cv2.putText(annotated_frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.circle(annotated_frame, com_global, 5, (255, 0, 0), -1)

            time_now = time.time()
            if (time_now - last_save_time > 5 and current_person_count > 0) or (current_person_count > previous_person_count):
                filename = f"frame_{int(time_now)}.jpg"
                save_path = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(save_path, annotated_frame)
                print(f"--- Frame saved: {filename} ---")
                last_save_time = time_now
            previous_person_count = current_person_count

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Human Pose and Distance Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        if pipeline: pipeline.stop()
        elif 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"Resources cleaned up. Log saved to {log_filename}")

if __name__ == "__main__":
    main()
