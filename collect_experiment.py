"""Collect everything needed to report a training run in a paper into one folder.

Usage (after training finishes):
  python3 collect_experiment.py --log logs/train_v2_raw.log
Output: experiments/<date>_<run name>/  (see README.md inside for the contents)
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["train_yolov8n.py", "letterbox.py", "predict_yolov8n.py", "compare_test.py", "collect_experiment.py"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(device):
    import cv2
    import numpy
    import torch
    import ultralytics

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None

    env = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
        "ram_gb": round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1) if sys.platform == "darwin" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "device": str(device),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("Apple MPS" if device == "mps" else None),
    }
    freeze = sh([sys.executable, "-m", "pip", "freeze"]) or ""
    return env, freeze


def dataset_stats(ds, names):
    from PIL import Image

    meta = yaml.safe_load((ds / "data.yaml").read_text()).get("roboflow", {})
    stats, manifest = {"roboflow": meta, "splits": {}}, []
    for folder in ("train", "valid", "test"):
        img_dir = ds / folder / "images"
        if not img_dir.is_dir():
            continue
        counts, widths, heights, empty, box_areas = [0] * len(names), [], [], 0, []
        images = sorted(img_dir.iterdir())
        for p in images:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
            lab = ds / folder / "labels" / (p.stem + ".txt")
            rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
            empty += not rows
            for r in rows:
                counts[int(r[0])] += 1
                box_areas.append(float(r[3]) * float(r[4]))
            manifest.append((folder, p.name, sha256(p), sha256(lab)))
        stats["splits"][folder] = {
            "images": len(images),
            "background_images": empty,
            "instances": dict(zip(names, counts)),
            "width": {"min": min(widths), "median": statistics.median(widths), "max": max(widths)},
            "height": {"min": min(heights), "median": statistics.median(heights), "max": max(heights)},
            "relative_box_area": {"min": round(min(box_areas), 5), "median": round(statistics.median(box_areas), 5),
                                  "max": round(max(box_areas), 5)} if box_areas else None,
        }
    joined = "".join(f"{a}{b}{c}{d}" for a, b, c, d in manifest)
    stats["manifest_sha256"] = hashlib.sha256(joined.encode()).hexdigest()
    return stats, manifest


def evaluate(weights, data_yaml, split, imgsz, device, out_dir):
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, batch=8, device=device, workers=2,
                               conf=0.001, iou=0.7, plots=True, project=str(out_dir), name=f"eval_{split}",
                               exist_ok=True, verbose=False)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r = float(m.box.p[i]), float(m.box.r[i])
        per_class[names[int(c)]] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                                    "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}
    mp, mr = float(m.box.mp), float(m.box.mr)
    return {
        "split": split,
        "all": {"precision": mp, "recall": mr, "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
                "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 3) for k, v in m.speed.items()},
        "eval_settings": {"conf": 0.001, "iou_nms": 0.7, "imgsz": imgsz, "device": str(device)},
    }


def clean_log(raw):
    text = raw.replace("\r", "\n")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    keep = []
    for line in text.splitlines():
        if not line.strip() or "Warning" in line or "run_backward" in line:
            continue
        if re.search(r"[━─]", line) and "100%" not in line:  # drop partial progress-bar frames
            continue
        keep.append(line.rstrip())
    return "\n".join(keep) + "\n"


def tables(metrics, names):
    rows = []
    for split in ("val", "test"):
        m = metrics[split]
        for cls in ["all", *names]:
            d = m["all"] if cls == "all" else m["per_class"].get(cls)
            if d:
                rows.append((split, cls, d["precision"], d["recall"], d["f1"], d["mAP50"], d["mAP50-95"]))
    md = ["| Split | Class | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {s} | {c} | {p:.3f} | {r:.3f} | {f:.3f} | {a:.3f} | {b:.3f} |" for s, c, p, r, f, a, b in rows]
    tex = [r"\begin{tabular}{llccccc}", r"\toprule",
           r"Split & Class & Precision & Recall & F1 & mAP@0.5 & mAP@0.5:0.95 \\", r"\midrule"]
    tex += [f"{s} & {c} & {p:.3f} & {r:.3f} & {f:.3f} & {a:.3f} & {b:.3f} \\\\" for s, c, p, r, f, a, b in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    with_csv = [("split", "class", "precision", "recall", "f1", "mAP50", "mAP50-95"), *rows]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n", with_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs/detect/yolov8n_finetune")
    ap.add_argument("--source-data", type=Path, default=ROOT / "datasets/dataset-real-g3nyo-v4",
                    help="original Roboflow export (before letterbox)")
    ap.add_argument("--log", type=Path, required=True, help="raw stdout/stderr log of the training run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    run_args = yaml.safe_load((args.run / "args.yaml").read_text())
    imgsz = run_args["imgsz"]
    data_yaml = Path(run_args["data"])
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = (args.out or ROOT / "experiments" / f"{dt.date.today():%Y-%m-%d}_{args.run.name}").resolve()
    out.mkdir(parents=True, exist_ok=False)

    # 1. Raw run directory from Ultralytics (args.yaml, results.csv, plots, weights).
    shutil.copytree(args.run, out / "run")

    # 2. Training logs.
    raw = args.log.read_text(errors="replace")
    shutil.copy(args.log, out / "train_log_raw.txt")
    (out / "train_log.txt").write_text(clean_log(raw))
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    optim = re.search(r"optimizer:\s*(\w+)\(lr=([\d.e-]+), momentum=([\d.e-]+)\)", raw)
    stop = re.search(r"(?:Training stopped early|Stopping training early).*?Best results observed at epoch (\d+)", raw, re.S)
    hours = re.search(r"(\d+) epochs completed in ([\d.]+) hours", raw)

    # 3. Per-epoch history and best epoch.
    with open(args.run / "results.csv") as f:
        hist = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    best_row = max(hist, key=lambda r: float(r["metrics/mAP50-95(B)"]))

    # 4. Final evaluation of best.pt on val and test (same letterboxed data as training).
    best = args.run / "weights" / "best.pt"
    model = YOLO(str(best))
    names = list(model.names.values())
    metrics = {s: evaluate(best, data_yaml, s, imgsz, device, out) for s in ("val", "test")}

    # 5. Model complexity.
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(get_flops(model.model, imgsz), 2)
    except Exception:
        gflops = None

    # 6. Dataset description + file manifest (for exact reproducibility).
    ds_stats, manifest = dataset_stats(args.source_data, names)
    with open(out / "dataset_manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([("split", "image", "image_sha256", "label_sha256"), *manifest])

    env, freeze = environment(device)
    (out / "requirements-freeze.txt").write_text(freeze + "\n")
    (out / "scripts").mkdir()
    for s in SCRIPTS:
        if (ROOT / s).exists():
            shutil.copy(ROOT / s, out / "scripts" / s)

    summary = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": {"architecture": "YOLOv8n", "init_weights": run_args["model"], "classes": names,
                  "parameters": n_params, "gflops_at_imgsz": gflops,
                  "best_pt_mb": round(best.stat().st_size / 2**20, 2), "best_pt_sha256": sha256(best),
                  "last_pt_sha256": sha256(args.run / "weights" / "last.pt")},
        "training": {"epochs_max": run_args["epochs"], "epochs_run": len(hist), "best_epoch": int(float(best_row["epoch"])),
                     "early_stopped": bool(stop), "patience": run_args["patience"],
                     "train_time_hours": float(hours[2]) if hours else None,
                     "optimizer_resolved": {"name": optim[1], "lr": float(optim[2]), "momentum": float(optim[3])} if optim else None,
                     "imgsz": imgsz, "batch": run_args["batch"], "seed": run_args["seed"],
                     "deterministic": run_args["deterministic"], "amp": run_args["amp"],
                     "preprocessing": f"letterbox to {imgsz}x{imgsz}: longer side scaled to {imgsz} (aspect ratio "
                                      "preserved, INTER_AREA down / INTER_LINEAR up), centered gray (114) padding",
                     "augmentation": {k: run_args[k] for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                                      "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
                                      "copy_paste", "close_mosaic") if k in run_args},
                     "hyperparameters": {k: run_args[k] for k in ("optimizer", "lr0", "lrf", "momentum", "weight_decay",
                                         "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
                                         "cos_lr", "iou") if k in run_args}},
        "dataset": ds_stats,
        "metrics": metrics,
        "environment": env,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md, tex, rows = tables(metrics, names)
    (out / "results_table.md").write_text(md)
    (out / "results_table.tex").write_text(tex)
    with open(out / "results_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    sp = ds_stats["splits"]
    (out / "README.md").write_text(f"""# Experiment: {args.run.name}

## Contents
- `summary.json`: all settings, dataset statistics, metrics, environment (machine-readable)
- `results_table.md` / `.tex` / `.csv`: final metrics of best.pt on val and test, per class
- `run/`: Ultralytics run directory: `args.yaml` (full config), `results.csv` (per-epoch losses and metrics),
  plots (`results.png`, PR/F1/P/R curves, confusion matrices, label stats), `weights/best.pt`, `weights/last.pt`
- `eval_val/`, `eval_test/`: plots from the final evaluation of best.pt
- `train_log.txt` (cleaned) and `train_log_raw.txt` (complete stdout/stderr)
- `dataset_manifest.csv`: SHA-256 of every image and label file used
- `requirements-freeze.txt`, `scripts/`: exact package versions and code

## Setup (draft text for the Methods section)
A COCO-pretrained YOLOv8n ({n_params:,} parameters{f", {gflops} GFLOPs" if gflops else ""} at {imgsz}x{imgsz}) was fine-tuned on a
{len(names)}-class dataset ({", ".join(names)}) exported from Roboflow
({ds_stats['roboflow'].get('workspace')}/{ds_stats['roboflow'].get('project')}, version {ds_stats['roboflow'].get('version')}).
The dataset contains {sum(s['images'] for s in sp.values())} images split into
{sp['train']['images']} train / {sp['valid']['images']} validation / {sp.get('test', {}).get('images', 0)} test.
All images were letterboxed to {imgsz}x{imgsz} (aspect ratio preserved, gray padding). Training ran for at most
{run_args['epochs']} epochs with early-stopping patience {run_args['patience']}, batch size {run_args['batch']}, seed {run_args['seed']},
using Ultralytics {env['ultralytics']} / PyTorch {env['torch']} on {env['cpu']} ({env['device']}).
Training stopped after {len(hist)} epochs; the best validation mAP@0.5:0.95 was reached at epoch {int(float(best_row['epoch']))}.
Metrics were computed with conf=0.001 and NMS IoU=0.7.

## Results (best.pt)
{md}
""")
    print(f"Experiment record written to {out}")
    print(md)


if __name__ == "__main__":
    main()
