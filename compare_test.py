"""Side-by-side ground truth vs prediction for a dataset split, using the training letterbox.

Usage:
  python3 compare_test.py                      # test split of datasets/dataset-real-g3nyo-v4
  python3 compare_test.py --split valid --conf 0.25
Writes one compare_<name>.jpg per image plus contact sheets (sheet_N.jpg) to --out.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from letterbox import letterbox, labels_to_letterbox

ROOT = Path(__file__).resolve().parent
COLORS = [(0, 200, 0), (255, 140, 0)]


def draw(img, boxes, names, title):
    out = img.copy()
    for c, x1, y1, x2, y2, label in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c % len(COLORS)], 2)
        cv2.putText(out, label, (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c % len(COLORS)], 2)
    if not boxes:
        cv2.putText(out, "no boxes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, title, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/dataset-real-g3nyo-v4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", type=Path, default=ROOT / "models/model_X3.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)  # tuned on valid with tune_threshold.py
    ap.add_argument("--iou", type=float, default=0.3, help="NMS IoU (tuned on valid)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/detect/test_compare")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(args.model))
    names = model.names
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, tally = [], {"gt": [0] * len(names), "pred": [0] * len(names)}
    for p in sorted((args.data / args.split / "images").iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxed, scale, px, py = letterbox(img, args.imgsz)
        rows = [tuple(map(float, r.split())) for r in
                (args.data / args.split / "labels" / (p.stem + ".txt")).read_text().splitlines() if r.strip()]
        s = args.imgsz
        gt = [(c, (x - bw / 2) * s, (y - bh / 2) * s, (x + bw / 2) * s, (y + bh / 2) * s, names[c])
              for c, x, y, bw, bh in labels_to_letterbox(rows, w, h, scale, px, py, s)]
        r = model.predict(boxed, imgsz=s, conf=args.conf, iou=args.iou, device=device, verbose=False)[0]
        pred = [(int(c), *b, f"{names[int(c)]} {cf:.2f}") for b, c, cf in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        for g in gt:
            tally["gt"][g[0]] += 1
        for q in pred:
            tally["pred"][q[0]] += 1
        pair = np.hstack([draw(boxed, gt, names, "GROUND TRUTH"), np.full((s, 6, 3), 255, np.uint8),
                          draw(boxed, pred, names, "PREDICTION")])
        cv2.imwrite(str(args.out / f"compare_{p.stem[:40]}.jpg"), pair)
        pairs.append(cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2)))
        print(f"{p.stem[:24]}  GT: {[g[5] for g in gt]}  PRED: {[q[5] for q in pred]}")
    per_sheet, cols = 12, 3
    for i in range(0, len(pairs), per_sheet):
        chunk = pairs[i:i + per_sheet]
        chunk += [np.full_like(pairs[0], 255)] * (-len(chunk) % cols)
        grid = np.vstack([np.hstack(chunk[j:j + cols]) for j in range(0, len(chunk), cols)])
        cv2.imwrite(str(args.out / f"sheet_{i // per_sheet + 1}.jpg"), grid)
    print("instances  GT:", dict(zip(names.values(), tally["gt"])), " PRED:", dict(zip(names.values(), tally["pred"])))
    print(f"Saved {len(pairs)} comparisons and contact sheets to {args.out}")


if __name__ == "__main__":
    main()
