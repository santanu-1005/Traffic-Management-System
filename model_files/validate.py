import multiprocessing
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train6/weights/best.pt")
    model.val(data="data.yaml")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()