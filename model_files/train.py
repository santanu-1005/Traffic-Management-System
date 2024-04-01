import multiprocessing
from ultralytics import YOLO

def main():
    model = YOLO()
    model.train(data="data.yaml", epochs=5)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()