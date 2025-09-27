from ultralytics import YOLO

def main():
    model = YOLO(r"C:/Users/20231807/Documents/GitHub/Cap_coral_reefs/New_version_of_the_project/yolov8_sm_latest.pt")
    model.train(
        data=r"C:/Users/20231807/Documents/GitHub/Cap_coral_reefs/New_version_of_the_project/data.yaml",
        imgsz=640, epochs=1, batch=2, seed=0, freeze=10,
        workers=2,          # you can raise later; set 0 if you still see issues
        device=0
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()     # required on Windows with spawn
    main()