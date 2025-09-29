# eval_collect.py
# Evaluate YOLOv8-seg runs (val/test) and write per-class + overall mAP to a CSV.
# Default paths assume this file lives in New_version_of_the_project/ next to:
#   - data.yaml
#   - runs/ (contains e.g., baseline_s0/weights/best.pt)

import argparse, os, glob
import pandas as pd
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data.yaml", help="Path to data.yaml")
    ap.add_argument("--runs",  default="runs",      help="A run dir OR a parent dir containing multiple runs")
    ap.add_argument("--out",   default="metrics_summary.csv", help="Output CSV")
    ap.add_argument("--split", default="test", choices=["val", "test"], help="Which split to evaluate on")
    ap.add_argument("--plots", action="store_true", help="Also save PR/F1 plots during val")
    return ap.parse_args()

def list_run_dirs(root: str):
    root = os.path.abspath(root)
    # Case 1: a single run folder with weights/best.pt
    if os.path.isdir(os.path.join(root, "weights")) and os.path.exists(os.path.join(root, "weights", "best.pt")):
        return [root]
    # Case 2: parent folder containing run subfolders
    kids = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    runs = [p for p in kids if os.path.exists(os.path.join(p, "weights", "best.pt"))]
    if not runs:
        raise SystemExit(f"No run folders with weights/best.pt found under: {root}")
    return sorted(runs)

def try_get_seg_metrics(m):
    out = {}
    seg = getattr(m, "seg", None)
    if seg is not None:
        out["mAP50-95_all"] = float(getattr(seg, "map", float("nan")))
        out["mAP50_all"]    = float(getattr(seg, "map50", float("nan")))
        out["per_class"]    = list(getattr(seg, "maps", []))  # per-class mAP50-95
        return out
    rd = getattr(m, "results_dict", None) or {}
    for k in ("metrics/seg_mAP50-95", "metrics/seg_map50-95", "metrics/mAP50-95(M)"):
        if k in rd: out["mAP50-95_all"] = float(rd[k]); break
    for k in ("metrics/seg_mAP50", "metrics/seg_map50", "metrics/mAP50(M)"):
        if k in rd: out["mAP50_all"] = float(rd[k]); break
    out["per_class"] = []
    return out

def main():
    args = parse_args()
    runs = list_run_dirs(args.runs)

    # Get class names from any best.pt (dict id->name)
    any_best = os.path.join(runs[0], "weights", "best.pt")
    names = YOLO(any_best).names
    if isinstance(names, dict):
        class_names = [names[i] for i in range(len(names))]
    else:
        class_names = list(names)

    rows = []
    for rd in runs:
        best = os.path.join(rd, "weights", "best.pt")
        run_name = os.path.basename(rd)
        model = YOLO(best)
        metrics = model.val(data=args.data, split=args.split, plots=args.plots)
        seg = try_get_seg_metrics(metrics)

        row = {
            "run": run_name,
            "split": args.split,
            "mAP50-95_all": seg.get("mAP50-95_all", float("nan")),
            "mAP50_all": seg.get("mAP50_all", float("nan")),
        }
        per = seg.get("per_class", [])
        for i, cname in enumerate(class_names):
            row[f"mAP50-95_{cname}"] = float(per[i]) if i < len(per) else float("nan")

        # Convenience: Macro-AP (average of the two classes if both present)
        if len(class_names) >= 2:
            a = row.get(f"mAP50-95_{class_names[0]}", float("nan"))
            b = row.get(f"mAP50-95_{class_names[1]}", float("nan"))
            try:
                row["MacroAP50-95"] = (a + b) / 2.0
            except Exception:
                row["MacroAP50-95"] = float("nan")

        rows.append(row)
        print(f"Evaluated {run_name}: mAP50-95_all={row['mAP50-95_all']:.3f}")

    df = pd.DataFrame(rows).sort_values(["run", "split"])
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")

if __name__ == "__main__":
    main()
