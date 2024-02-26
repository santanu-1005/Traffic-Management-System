import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

ZONE_POLYGON = np.array([
    [0,0],
    [1280,0],
    [1250, 720],
    [0, 720]
])

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    frame_width, frame_height = args.webcam_resolution
     
    cap = cv2.VideoCapture("test.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )

    zone = sv.PolygonZone(polygon = ZONE_POLYGON, frame_resolution_wh = tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone = zone, color = sv.Color.red())

    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)  

        frame = box_annotator.annotate(scene = frame, detections = detections) 
        zone.trigger(detections = detections)
        frame = zone_annotator.annotate(scene = frame)
        cv2.imshow("yolov8",frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == '__main__':
    main()