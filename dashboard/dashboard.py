import argparse
import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import numpy as np


ZONE_POLYGON = np.array([
    [0,0],
    [900,0],
    [900, 880],
    [0, 880]
])


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1080, 720],
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

model = YOLO("model_files/yolov8n.pt")
box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 2,
    text_scale = 1
)

args = parse_arg()

zone = sv.PolygonZone(polygon = ZONE_POLYGON, frame_resolution_wh = tuple(args.webcam_resolution))
zone_annotator = sv.PolygonZoneAnnotator(zone = zone, color = sv.Color.RED)


st.title("Intelligent Traffic Management System")

# Left section - Project details
st.subheader("Project Details")
total_vehicles = 0  # Replace with actual counter
car_count = 0  # Replace with actual counter
# Add more details as needed
st.text(f"Project Title: Your Project Name")
st.text(f"Total Vehicles Detected: {total_vehicles}")
st.text(f"Number of Cars Detected: {car_count}")

# Right section - Video streams
col1, col2 = st.columns(2)

# Capture video stream
cap = cv2.VideoCapture("test.mp4")  # Change 0 to video file path if needed

with col1:
    st.subheader("Raw Video")
    video1 = open('test.mp4', "rb")
    video1_bytes = video1.read()
    st.video(video1_bytes, start_time=0)

with col2:
    st.subheader("Video with Detection")

    video2 = open('test.mp4', "rb")
    video2_bytes = video2.read()
    st.video(video2_bytes, start_time=0)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     result = model(frame)[0]
    #     detections = sv.Detections.from_ultralytics(result)  

    #     frame = box_annotator.annotate(scene = frame, detections = detections) 
    #     zone.trigger(detections = detections)
    #     frame = zone_annotator.annotate(scene = frame)

        # # Perform object detection
        # detected_objects = detect_objects(frame)
        # # Draw bounding boxes (replace with your logic)
        # for obj in detected_objects:
        #     x1, y1, x2, y2, label, confidence = obj
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # video_frame(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cap.release()
cv2.destroyAllWindows()
