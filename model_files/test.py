import multiprocessing
from ultralytics import YOLO

def main():
    model = YOLO("model_files/yolov8n.pt")
    model.predict(source="test.mp4")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()