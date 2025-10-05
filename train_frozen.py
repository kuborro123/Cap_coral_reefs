from ultralytics import YOLO

def main():
    # === Load pretrained YOLOv8 segmentation model ===
    model = YOLO(r"C:\Users\User\Desktop\YEAR 3\Q1\Capstone DC\yolov8_sm_latest.pt")

    # === Freeze backbone layers (first 10 layers, adjust if needed) ===
    model.freeze = 10

    # === Train with validation happening automatically ===
    results = model.train(
        data=r"C:\Users\User\Downloads\website\website\benthic_datasets\mask_labels\reef_support\out_fold\data.yaml",
        epochs=2,       # number of epochs
        batch=8,         # batch size
        imgsz=640,       # image size
        name="reef_frozen_train",   # run folder name
        workers=0        # safer on Windows
    )

    print("✅ Training complete. Results (weights, metrics) saved at:", results.save_dir)

if __name__ == "__main__":
    main()
