from ultralytics import YOLO

def main():
    # === Load the best trained model ===
    model = YOLO(r"runs/segment/reef_frozen_train/weights/best.pt")

    # === Evaluate on test set (defined in data.yaml) ===
    metrics = model.val(
        data=r"C:\Users\User\Desktop\YEAR 3\Q1\Capstone DC\Datasets\mask_labels\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\benthic_datasets\mask_labels\reef_support\data.yaml",
        split="test"  # force using the test set
    )

    print("✅ Final test metrics:", metrics)

if __name__ == "__main__":
    main()
