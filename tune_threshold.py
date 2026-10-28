"""Sweep confidence and NMS-IoU thresholds on one split and report box counts, P, R, F1.

A prediction is a true positive if it matches an unmatched ground-truth box of the same class with IoU >= 0.5.
Tune on --split valid, then confirm the chosen values once on --split test.

Usage:
  python3 tune_threshold.py --split valid
  python3 tune_threshold.py --split test --conf 0.4 --iou 0.5
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from letterbox import letterbox, labels_to_letterbox

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def score(samples, conf, nc):
    tp, fp, fn = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    for gt, pred in samples:
        keep = pred[pred[:, 5] >= conf]
        keep = keep[np.argsort(-keep[:, 5])]
        for c in range(nc):
            g = gt[gt[:, 0] == c][:, 1:5]
            used = np.zeros(len(g), bool)
            for p in keep[keep[:, 4] == c]:
                if len(g):
                    ious = iou(p[:4], g)
                    ious[used] = 0
                    j = ious.argmax()
                    if ious[j] >= 0.5:
                        used[j] = True
                        tp[c] += 1
                        continue
                fp[c] += 1
            fn[c] += (~used).sum()
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/dataset-real-g3nyo-v4")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--model", type=Path, default=ROOT / "models/model_X3.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, nargs="*", default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    ap.add_argument("--iou", type=float, nargs="*", default=[0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(args.model))
    names = model.names
    nc = len(names)
    images = []
    for p in sorted((args.data / args.split / "images").iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxed, scale, px, py = letterbox(img, args.imgsz)
        rows = [tuple(map(float, r.split())) for r in
                (args.data / args.split / "labels" / (p.stem + ".txt")).read_text().splitlines() if r.strip()]
        s = args.imgsz
        gt = np.array([(c, (x - bw / 2) * s, (y - bh / 2) * s, (x + bw / 2) * s, (y + bh / 2) * s)
                       for c, x, y, bw, bh in labels_to_letterbox(rows, w, h, scale, px, py, s)]).reshape(-1, 5)
        images.append((boxed, gt))
    n_gt = sum(len(g) for _, g in images)
    print(f"{args.split}: {len(images)} images, {n_gt} ground-truth boxes")
    print(f"{'iou':>5} {'conf':>5} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}  per-class F1")
    best = None
    for nms_iou in args.iou:
        samples = []
        for boxed, gt in images:
            r = model.predict(boxed, imgsz=args.imgsz, conf=min(args.conf), iou=nms_iou, device=device, verbose=False)[0]
            pred = np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()[:, None], r.boxes.conf.cpu().numpy()[:, None]])
            samples.append((gt, pred))
        for conf in args.conf:
            tp, fp, fn = score(samples, conf, nc)
            T, F, N = tp.sum(), fp.sum(), fn.sum()
            p, r = T / max(T + F, 1), T / max(T + N, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            pcf = " ".join(f"{names[c]}={2 * tp[c] / max(2 * tp[c] + fp[c] + fn[c], 1):.3f}" for c in range(nc))
            print(f"{nms_iou:>5.2f} {conf:>5.2f} {T + F:>5} {T:>4} {F:>4} {N:>4} {p:>6.3f} {r:>6.3f} {f1:>6.3f}  {pcf}")
            if best is None or f1 > best[0]:
                best = (f1, nms_iou, conf, T + F)
    print(f"\nBest F1 {best[0]:.3f} at iou={best[1]} conf={best[2]} ({best[3]} predicted boxes vs {n_gt} ground truth)")


if __name__ == "__main__":
    main()
