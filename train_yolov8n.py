"""Fine-tune pretrained YOLOv8n on the Roboflow dataset.

Settings are read from ./.env (never printed):
  ROBOFLOW_API_KEY     required to download the dataset
  ROBOFLOW_WORKSPACE   e.g. detect-object-under-water
  ROBOFLOW_PROJECT     e.g. dataset-real-g3nyo
  ROBOFLOW_VERSION     e.g. 1
CLI flags override the non-secret settings. Use --data to skip the download.

Usage:
  python3 train_yolov8n.py                       # download (if needed) + train
  python3 train_yolov8n.py --data datasets/dataset-real-g3nyo-v4
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(path):
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
            if m:
                env[m[1]] = m[2].strip("\"'")
    return env


def use_certifi_if_needed():
    # python.org builds on macOS ship without a CA bundle; keep TLS verification on via certifi.
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


def download_dataset(key, workspace, project, version, out_dir):
    if out_dir.exists():
        print(f"Dataset already present: {out_dir}")
        return out_dir
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?" + urllib.parse.urlencode({"api_key": key})
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            link = json.load(r)["export"]["link"]
    except urllib.error.HTTPError as e:
        sys.exit(f"Roboflow export request failed (HTTP {e.code}); check workspace/project/version and API key.")
    out_dir.mkdir(parents=True)
    zip_path = out_dir.with_suffix(".zip")
    urllib.request.urlretrieve(link, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name.split("/"):
                sys.exit(f"Unsafe path in archive: {name}")
        zf.extractall(out_dir)
    zip_path.unlink()
    print(f"Downloaded dataset to {out_dir}")
    return out_dir


def check_dataset(ds):
    import yaml

    meta = yaml.safe_load((ds / "data.yaml").read_text())
    names = meta["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    nc = len(names)
    print(f"Classes ({nc}): {names}")
    splits = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        img_dir = ds / folder / "images"
        if not img_dir.is_dir():
            if split != "test":
                sys.exit(f"Missing required split: {img_dir}")
            continue
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        counts, problems = [0] * nc, 0
        for img in images:
            lab = ds / folder / "labels" / (img.stem + ".txt")
            if not lab.exists():
                problems += 1
                continue
            for row in (r.split() for r in lab.read_text().splitlines() if r.strip()):
                c = int(row[0])
                if len(row) != 5 or not 0 <= c < nc or not all(0 <= float(v) <= 1 for v in row[1:]):
                    problems += 1
                else:
                    counts[c] += 1
        print(f"  {split}: {len(images)} images, instances {dict(zip(names, counts))}, label problems: {problems}")
        if problems:
            sys.exit(f"Fix label problems in {folder} before training.")
        splits[split] = str(img_dir)
    # Roboflow's data.yaml uses '../train/images' style paths; write absolute paths instead.
    fixed = ds / "data_abs.yaml"
    fixed.write_text(yaml.safe_dump({**splits, "nc": nc, "names": names}, sort_keys=False))
    return fixed


def letterbox_dataset(ds, size):
    """Write a copy of the dataset with every image letterboxed to size x size and labels remapped."""
    import cv2

    from letterbox import labels_to_letterbox, letterbox

    out = ds.parent / f"{ds.name}-lb{size}"
    if (out / "data.yaml").exists():
        print(f"Letterboxed dataset already present: {out}")
        return out
    for folder in ("train", "valid", "test"):
        img_dir = ds / folder / "images"
        if not img_dir.is_dir():
            continue
        (out / folder / "images").mkdir(parents=True, exist_ok=True)
        (out / folder / "labels").mkdir(parents=True, exist_ok=True)
        n = 0
        for img_path in sorted(img_dir.iterdir()):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxed, scale, px, py = letterbox(img, size)
            cv2.imwrite(str(out / folder / "images" / (img_path.stem + ".jpg")), boxed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows = [tuple(map(float, r.split())) for r in
                    (ds / folder / "labels" / (img_path.stem + ".txt")).read_text().splitlines() if r.strip()]
            lines = [f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}" for c, x, y, bw, bh in
                     labels_to_letterbox(rows, w, h, scale, px, py, size)]
            (out / folder / "labels" / (img_path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
        print(f"  letterboxed {folder}: {n} images -> {size}x{size}")
    (out / "data.yaml").write_text((ds / "data.yaml").read_text())
    return out


def pick_device_and_batch(fixed_batch):
    import torch

    if torch.cuda.is_available():
        return 0, -1  # AutoBatch (~60% of CUDA memory)
    if torch.backends.mps.is_available():
        return "mps", fixed_batch
    return "cpu", fixed_batch


def main():
    env = load_env(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=env.get("ROBOFLOW_WORKSPACE"))
    ap.add_argument("--project", default=env.get("ROBOFLOW_PROJECT"))
    ap.add_argument("--version", default=env.get("ROBOFLOW_VERSION"))
    ap.add_argument("--data", type=Path, help="existing YOLOv8 export directory (skips download)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="fixed batch for CPU/MPS (CUDA uses AutoBatch)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true", help="enable RAM image caching (off by default)")
    ap.add_argument("--name", default="yolov8n_finetune")
    ap.add_argument("--weights", default="yolov8n.pt", help="starting weights (default: COCO-pretrained yolov8n.pt)")
    args = ap.parse_args()

    use_certifi_if_needed()
    if args.data:
        ds = args.data.resolve()
    else:
        missing = [n for n, v in (("ROBOFLOW_API_KEY", env.get("ROBOFLOW_API_KEY")),
                                  ("ROBOFLOW_WORKSPACE", args.workspace),
                                  ("ROBOFLOW_PROJECT", args.project),
                                  ("ROBOFLOW_VERSION", args.version)) if not v]
        if missing:
            sys.exit(f"Missing settings in .env (or CLI flags): {', '.join(missing)}")
        ds = download_dataset(env["ROBOFLOW_API_KEY"], args.workspace, args.project, args.version,
                              ROOT / "datasets" / f"{args.project}-v{args.version}")
    check_dataset(ds)
    # Explicit resize: letterbox every image to imgsz x imgsz (aspect preserved, gray padding).
    data_yaml = check_dataset(letterbox_dataset(ds, args.imgsz))

    from ultralytics import YOLO

    device, batch = pick_device_and_batch(args.batch)
    print(f"Device: {device}, batch: {'auto' if batch == -1 else batch}, workers: {args.workers}, cache: {args.cache}")
    model = YOLO(args.weights)  # COCO-pretrained yolov8n.pt unless --weights is given
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,  # letterbox: aspect-preserving resize + pad
        batch=batch,
        device=device,
        workers=args.workers,
        cache=args.cache,
        pretrained=True,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=False,
        save=True,
        plots=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"

    metrics = YOLO(str(best)).val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=batch if batch > 0 else 16,
                                  device=device, workers=args.workers, project=str(save_dir), name="val_best")
    print("\n=== Results ===")
    print(f"best.pt: {best}")
    print(f"last.pt: {last}")
    print(f"val mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}  "
          f"precision: {metrics.box.mp:.4f}  recall: {metrics.box.mr:.4f}")
    names = metrics.names
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"  {names[int(c)]}: mAP50-95 {metrics.box.maps[int(c)]:.4f}  mAP50 {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
