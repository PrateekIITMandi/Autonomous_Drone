"""
Human Pose, 3D Distance Estimation, and Vehicle Direction (Robust Geolocation Version)

This script performs real-time human detection, calculates the 3D position,
and computes the subject's real-world GPS coordinates by fusing data from a
YOLOv8 model, a depth camera, and a Pixhawk flight controller with GPS.

Features:
- Geolocation: Calculates and displays the GPS coordinates of detected humans.
- Outlier rejection: Skips detections where AI and sensor distances differ significantly.
- Tiered AI Distance Calibration: Applies a multi-level subtraction model to the AI distance.
- Background threading for non-blocking, real-time Pixhawk GPS and Attitude data.
- Detailed terminal output and comprehensive CSV logging with all data points.
"""
from ultralytics import YOLO
import cv2
import time
import numpy as np
import math
import os
import csv
from datetime import datetime
import threading
from pymavlink import mavutil

# --- Global variables for Pixhawk data ---
current_yaw_deg = 0.0
current_lat = 0.0
current_lon = 0.0
is_pixhawk_connected = False

# --- Helper Functions ---

def calculate_com(points):
    """Calculates the Center of Mass from a list of keypoints."""
    valid_points = [p for p in points if p[0] > 0 and p[1] > 0]
    if not valid_points: return None
    return tuple(np.mean(np.array(valid_points), axis=0).astype(int))

def estimate_distance(known_width, focal_length, pixel_width):
    """Estimates Z distance based on a known width, focal length, and pixel width."""
    if pixel_width < 1e-5: return 0
    return (known_width * focal_length) / pixel_width

def calculate_orientation(p1, p2):
    """Calculates the angle of the line connecting two points with the horizontal axis."""
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return math.degrees(angle_rad), angle_rad

def get_human_gps_location(lat_deg, lon_deg, bearing_deg, distance_m):
    """
    Calculates the GPS coordinates of a point given a starting point,
    bearing, and distance.
    """
    if distance_m <= 0:
        return lat_deg, lon_deg

    R = 6378137.0  # Earth's radius in meters
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)
    
    d_R = distance_m / R
    
    lat2_rad = math.asin(math.sin(lat_rad) * math.cos(d_R) +
                        math.cos(lat_rad) * math.sin(d_R) * math.cos(bearing_rad))
    
    lon2_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(d_R) * math.cos(lat_rad),
                                     math.cos(d_R) - math.sin(lat_rad) * math.sin(lat2_rad))
    
    return math.degrees(lat2_rad), math.degrees(lon2_rad)

# --- Pixhawk Data Fetching Thread ---
def pixhawk_thread_func(port):
    """
    Connects to Pixhawk and continuously updates global GPS and attitude variables.
    """
    global current_yaw_deg, is_pixhawk_connected, current_lat, current_lon
    while True:
        try:
            print(f"Attempting to connect to Pixhawk on {port}...")
            master = mavutil.mavlink_connection(port, baud=9600)
            master.wait_heartbeat()
            print("Pixhawk Heartbeat received! Streaming data.")
            is_pixhawk_connected = True
            while True:
                # Listen for both GPS and Attitude messages
                msg = master.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True)
                if msg:
                    if msg.get_type() == 'ATTITUDE':
                        yaw_deg = math.degrees(msg.yaw)
                        current_yaw_deg = yaw_deg + 360 if yaw_deg < 0 else yaw_deg
                    elif msg.get_type() == 'GLOBAL_POSITION_INT':
                        current_lat = msg.lat / 1e7
                        current_lon = msg.lon / 1e7
        except Exception as e:
            print(f"Pixhawk connection failed or lost: {e}")
            is_pixhawk_connected = False
            time.sleep(5)

# --- Main Function ---
def main():
    # --- Configuration ---
    USE_REALSENSE = True
    USE_PIXHAWK = True
    PIXHAWK_PORT = '/dev/ttyACM0'  # <-- CHANGE THIS
    CONFIDENCE_THRESHOLD = 0.80
    MODEL_PATH = "yolov8l-pose.onnx"
    SAVE_DIR = "saved_frames"
    MAX_SENSOR_DEPTH = 14.0
    AI_OUTLIER_THRESHOLD_M = 2.0

    os.makedirs(SAVE_DIR, exist_ok=True)
    AVG_SHOULDER_WIDTH = 0.387

    if USE_PIXHAWK:
        threading.Thread(target=pixhawk_thread_func, args=(PIXHAWK_PORT,), daemon=True).start()

    print(f"Loading optimized model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_filename, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        header = ["Timestamp", "Person_ID", "Calibrated_Distance_m", "Sensor_Distance_m", "Original_AI_Distance_m", "Orientation_deg", "Coord_X_m", "Coord_Y_m", "Direction_deg", "Human_Lat", "Human_Lon"]
        csv_writer.writerow(header)
        print(f"Logging data to {log_filename}")

        pipeline = None
        try:
            if USE_REALSENSE:
                import pyrealsense2 as rs
                print("Initializing Intel RealSense Camera...")
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                profile = pipeline.start(config)
                align = rs.align(rs.stream.color)
                intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                FOCAL_LENGTH, FRAME_CENTER_X, FRAME_CENTER_Y = intrinsics.fx, intrinsics.ppx, intrinsics.ppy
            else:
                raise RuntimeError("Webcam mode requires manual calibration values.")
        except Exception as e:
            print(f"Error: {e}. This script requires a RealSense camera.")
            return

        prev_time = time.time()
        last_save_time = time.time()
        previous_person_count = 0

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()
                if not color_frame_rs or not depth_frame_rs: continue
                frame = np.asanyarray(color_frame_rs.get_data())
                annotated_frame = frame.copy()
                
                results = model(frame, verbose=False)
                current_person_count = 0
                
                for result in results:
                    if result.boxes and result.boxes.conf is not None:
                        for i in range(len(result.boxes)):
                            if result.boxes.conf[i] >= CONFIDENCE_THRESHOLD:
                                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = box
                                
                                if result.keypoints and len(result.keypoints.xy) > i:
                                    keypoints = result.keypoints.xy.cpu().numpy()[i]
                                    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
                                    com_global = calculate_com(keypoints)

                                    if (l_shoulder[0] > 0 and r_shoulder[0] > 0) and com_global:
                                        orientation_deg, orientation_rad = calculate_orientation(l_shoulder, r_shoulder)
                                        pixel_shoulder_width = math.dist(l_shoulder, r_shoulder)
                                        corrected_shoulder_width = AVG_SHOULDER_WIDTH * abs(math.cos(orientation_rad))
                                        
                                        original_ai_z = estimate_distance(corrected_shoulder_width, FOCAL_LENGTH, pixel_shoulder_width)
                                        sensor_depth_m = depth_frame_rs.get_distance(com_global[0], com_global[1])
                                        if sensor_depth_m > MAX_SENSOR_DEPTH or sensor_depth_m == 0:
                                            sensor_depth_m = 0

                                        if sensor_depth_m > 0 and abs(original_ai_z - sensor_depth_m) > AI_OUTLIER_THRESHOLD_M:
                                            continue
                                        
                                        calibrated_ai_z = original_ai_z
                                        if 0 < original_ai_z < 2.5: calibrated_ai_z -= 0.3
                                        elif 2.5 <= original_ai_z < 4.5: calibrated_ai_z -= 0.9
                                        elif original_ai_z >= 4.5: calibrated_ai_z -= 1.13
                                        
                                        current_person_count += 1
                                        direction = current_yaw_deg if is_pixhawk_connected else -1.0
                                        
                                        human_lat, human_lon = -1, -1
                                        if is_pixhawk_connected and current_lat != 0.0:
                                            human_lat, human_lon = get_human_gps_location(current_lat, current_lon, direction, calibrated_ai_z)

                                        # Log and Print Data
                                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                        log_data = [timestamp, current_person_count, f"{calibrated_ai_z:.3f}", f"{sensor_depth_m:.3f}", f"{original_ai_z:.3f}", f"{orientation_deg:.1f}", 0, 0, f"{direction:.2f}", f"{human_lat:.7f}", f"{human_lon:.7f}"]
                                        csv_writer.writerow(log_data)
                                        
                                        term_output = (f"Person {current_person_count}: Z:{calibrated_ai_z:.2f}m | Dir:{direction:.1f} | "
                                                       f"Human GPS: {human_lat:.6f}, {human_lon:.6f}")
                                        print(term_output)

                                        # Draw on Frame
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        info_text_1 = f"Dist: {calibrated_ai_z:.2f}m | Tilt: {orientation_deg:.1f}"
                                        info_text_2 = f"Dir: {direction:.1f} deg"
                                        info_text_3 = f"GPS: {human_lat:.5f}, {human_lon:.5f}"
                                        cv2.putText(annotated_frame, info_text_1, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, info_text_2, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, info_text_3, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        cv2.circle(annotated_frame, com_global, 5, (255, 0, 0), -1)

                time_now = time.time()
                if (time_now - last_save_time > 5 and current_person_count > 0) or (current_person_count > previous_person_count):
                    cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{int(time_now)}.jpg"), annotated_frame)
                    last_save_time = time_now
                previous_person_count = current_person_count

                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if USE_PIXHAWK:
                    status_text = f"Pixhawk: {'Connected' if is_pixhawk_connected else 'Disconnected'}"
                    status_color = (0, 255, 0) if is_pixhawk_connected else (0, 0, 255)
                    cv2.putText(annotated_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    gps_text = f"Vehicle GPS: {current_lat:.6f}, {current_lon:.6f}"
                    cv2.putText(annotated_frame, gps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                cv2.imshow("Human Geolocation", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            if pipeline: pipeline.stop()
            cv2.destroyAllWindows()
            print(f"Resources cleaned up. Log saved to {log_filename}")

if __name__ == "__main__":
    main()


