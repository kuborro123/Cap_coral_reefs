from ultralytics import YOLO

# 1. Load your already trained YOLO model (first-stage training weights)
model = YOLO(r"runs/segment/exp/weights/best.pt")   # change path to your best.pt

# 2. Fine-tune on the oversampled dataset
results = model.train(
    data=r"splits/data_soft100.yaml",  # points to train_soft_100to1.txt, val.txt, test.txt
    epochs=10,                         # keep short, it's fine-tuning
    imgsz=640,                         # image size
    lr0=0.0005,                        # smaller LR for fine-tune
    freeze=10,                         # freeze backbone, train head only
    name="finetune_soft100"            # run name (results go to runs/segment/finetune_soft100/)
)

# 3. Evaluate the fine-tuned model on the original balanced dataset
metrics = model.val(
    data=r"C:\Users\User\Downloads\website\website\benthic_datasets\mask_labels\reef_support\out_fold\data.yaml",  # your original dataset config (not oversampled)
    imgsz=640
)

print("Validation metrics:", metrics)
