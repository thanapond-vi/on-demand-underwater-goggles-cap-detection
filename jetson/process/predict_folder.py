"""
predict_folder.py - runs on the Jetson Nano (TensorRT, no Docker / ultralytics needed)

Predicts the GoPro photos stored in images/<session>/raw/ (sent there by the Pi Zero) when asked,
counts caps/glasses, and writes for each session:
    images/<session>/predicted/pred_<name>.jpg   image with boxes drawn
    images/<session>/results.jsonl               one line per photo: file, counts, timing (ms)
    images/<session>/compare/compare_<name>.jpg  ORIGINAL | PREDICTION side by side (like compare_test.py)
    images/<session>/sheet_N.jpg                 12 comparisons per sheet (only with --sheet)
raw/ keeps the GoPro photos untouched (full size, no boxes), e.g. for labelling and training later.
Photos already predicted are skipped, so running it again only does the new ones.

Usage:
    python3 predict_folder.py --list           # sessions and how many photos are not predicted yet
    python3 predict_folder.py test1            # predict session test1
    python3 predict_folder.py                  # predict every session with new photos
    python3 predict_folder.py test1 --redo     # predict all photos of test1 again
"""
import argparse
import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent   # folder holding process/, models/, images/
IMAGES = ROOT / "images"                        # one folder per session: raw/, predicted/, results.jsonl
IMG_EXT = (".jpg", ".jpeg", ".png")


def photos(session):
    return sorted(f for f in (IMAGES / session / "raw").iterdir() if f.suffix.lower() in IMG_EXT)


def todo(session, redo=False):
    out = IMAGES / session / "predicted"
    return [f for f in photos(session) if redo or not (out / ("pred_" + f.name)).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", help="session folders in images (default: all with new photos)")
    ap.add_argument("--model", default="model_X3", help="model name in models/ (model_X3: YOLOv8n fine-tuned on the pool photographs)")
    ap.add_argument("--engine", type=Path, help="TensorRT engine (default: models/<model>_fp16.engine)")
    ap.add_argument("--conf", type=float, default=0.35)  # tuned on valid with tune_threshold.py
    ap.add_argument("--iou", type=float, default=0.3, help="NMS IoU (tuned on valid)")
    ap.add_argument("--redo", action="store_true", help="predict photos again even if already done")
    ap.add_argument("--list", action="store_true", help="only list sessions")
    ap.add_argument("--sheet", action="store_true", help="also make sheet_N.jpg with 12 comparisons each")
    args = ap.parse_args()
    args.engine = args.engine or ROOT / "models" / (args.model + "_fp16.engine")

    IMAGES.mkdir(parents=True, exist_ok=True)
    all_sessions = sorted(d.name for d in IMAGES.iterdir() if (d / "raw").is_dir())
    if args.list:
        for s in all_sessions:
            print("{}: {} photo(s), {} not predicted".format(s, len(photos(s)), len(todo(s))))
        return
    for s in args.sessions:
        if s not in all_sessions:
            raise SystemExit("no session {} in {}".format(s, IMAGES))
    sessions = args.sessions or [s for s in all_sessions if todo(s)]
    jobs = [(s, todo(s, args.redo)) for s in sessions]
    if not any(files for _, files in jobs):
        print("nothing new to predict")
        return

    import compare
    from trt_yolo import NAMES, TrtModel, detect, draw  # load TensorRT only when there is work

    model = TrtModel(args.engine)  # includes warm-up
    for session, files in jobs:
        out = IMAGES / session / "predicted"
        out.mkdir(parents=True, exist_ok=True)
        cmp_dir = IMAGES / session / "compare"
        cmp_dir.mkdir(exist_ok=True)
        if args.redo and (IMAGES / session / "results.jsonl").exists():
            (IMAGES / session / "results.jsonl").unlink()  # all photos are redone, start the log fresh
        total = {n: 0 for n in NAMES}
        for f in files:
            t0 = time.perf_counter()
            img = cv2.imread(str(f))
            if img is None:
                print(json.dumps({"file": f.name, "error": "unreadable"}), flush=True)
                continue
            raw = img.copy()
            xyxy, cls, conf, speed = detect(model, img, args.conf, args.iou)
            cv2.imwrite(str(out / ("pred_" + f.name)), draw(img, xyxy, cls, conf))
            t1 = time.perf_counter()

            counts = {n: 0 for n in NAMES}
            for c in cls:
                counts[NAMES[c]] += 1
                total[NAMES[c]] += 1
            result = {
                "file": f.name,
                "counts": counts,
                "total_ms": round((t1 - t0) * 1000, 1),  # read + detect + draw/save for this photo
                "preprocess_ms": round(speed["preprocess"], 1),
                "inference_ms": round(speed["inference"], 1),
                "postprocess_ms": round(speed["postprocess"], 1),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            with open(str(IMAGES / session / "results.jsonl"), "a") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            confs = [(NAMES[c], float(cf)) for c, cf in zip(cls, conf)]
            cv2.imwrite(str(cmp_dir / ("compare_" + f.stem + ".jpg")), compare.pair(raw, img, counts, f.name, confs))
        if args.sheet:
            compare.sheets(cmp_dir, IMAGES / session)
        print("{}: {} photo(s) predicted, total {}".format(session, len(files), total), flush=True)
    model.close()


if __name__ == "__main__":
    main()
