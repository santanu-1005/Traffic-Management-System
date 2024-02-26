import multiprocessing
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train5/weights/best.pt")
    model.predict(source="test.mp4")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()