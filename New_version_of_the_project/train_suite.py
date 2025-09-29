# train_suite.py
import argparse, os, sys
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8_sm_latest.pt", help="Path to YOLOv8 seg model .pt")
    ap.add_argument("--data",  default="data.yaml", help="Path to base data.yaml (fixed val/test)")
    ap.add_argument("--runs",  default="runs", help="Directory to store runs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--batch",  default="auto", help='int/float or "auto"')
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds",  type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None, help="CUDA device index, e.g. 0; leave empty for auto")
    ap.add_argument("--names", nargs="+", default=["baseline:splits/train_soft_1to1.txt"],
                    help="Pairs name:train_list.txt (space separated)")
    return ap.parse_args()

def _normalize_batch(b):
    # Python API wants int/float; map "auto" -> -1 (AutoBatch)
    if isinstance(b, (int, float)):
        return b
    if isinstance(b, str):
        if b.lower() == "auto":
            return -1
        try:
            return int(b)
        except ValueError:
            return float(b)
    return b

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model .pt not found: {args.model}"); sys.exit(1)
    if not os.path.exists(args.data):
        print(f"[ERROR] data.yaml not found: {args.data}"); sys.exit(1)
    os.makedirs(args.runs, exist_ok=True)

    # parse name:path pairs
    pairs = {}
    for item in args.names:
        if ":" not in item:
            print(f"[ERROR] Bad --names item (expected name:path): {item}"); sys.exit(1)
        name, train_list = item.split(":", 1)
        if not os.path.exists(train_list):
            print(f"[ERROR] Train list not found: {train_list}"); sys.exit(1)
        pairs[name] = train_list

    # read base yaml
    with open(args.data, "r", encoding="utf-8") as f:
        base_yaml = f.read()

    batch_val = _normalize_batch(args.batch)

    for seed in args.seeds:
        for name, train_list in pairs.items():
            print(f"\n=== Starting training: {name} (seed {seed}) ===")
            # write temp yaml with train swapped
            tmp_yaml = os.path.join(os.path.dirname(args.data), f"data_{name}_s{seed}.yaml")
            lines, swapped = [], False
            for line in base_yaml.splitlines():
                if line.strip().lower().startswith("train:"):
                    lines.append(f"train: {train_list}"); swapped = True
                else:
                    lines.append(line)
            if not swapped:
                lines.append(f"train: {train_list}")
            with open(tmp_yaml, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            run_name = f"{name}_s{seed}"
            try:
                model = YOLO(args.model)
                model.train(
                    data=tmp_yaml,
                    imgsz=args.imgsz,
                    epochs=args.epochs,
                    batch=batch_val,      # <- now valid
                    freeze=args.freeze,
                    workers=args.workers, # set 0 if Windows spawn gives issues
                    seed=seed,
                    device=args.device,   # e.g., "0"
                    project=args.runs,
                    name=run_name
                )
                print(f"✓ Completed: {run_name}")
            except Exception as e:
                print(f"[ERROR] Training failed for {run_name}: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()
