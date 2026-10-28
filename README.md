# Underwater detection and counting of swim caps and goggles (YOLOv8n + Jetson Nano)

Code and trained model for the paper *On-Demand Underwater Detection and Counting of Swimming Goggles and Swim Caps Using Edge AI*
(P. Chansuksri, T. Visessang, P. Chumchu, Faculty of Engineering at Sriracha, Kasetsart University).

A GoPro HERO13 Black takes still photographs underwater, a Raspberry Pi Zero 2 W downloads them over Wi-Fi and forwards them
over LAN to a Jetson Nano, which detects and counts swim caps (`cap`) and goggles (`glasses`) with a YOLOv8n TensorRT engine
and sends the results to a computer.

## Dataset
Roboflow project `detect-object-under-water/dataset-real-g3nyo`, **version 4** (YOLOv8 format):
150 photographs taken on 18 October 2026 in the swimming pool of Kasetsart University, Sriracha Campus (1.2 m deep).

| Split | Source | Photos | Caps | Goggles |
|---|---|---|---|---|
| train | session 20261018_155420 (GP012132-GP012231) | 100 originals + 200 augmented | 496 (1479 with augmented) | 333 (993) |
| valid | session 20261018_154604 (GP012082-GP012109) | 28 | 149 | 107 |
| test | session 20261018_154604 (GP012110-GP012131) | 22 | 126 | 73 |

Roboflow settings: Auto-Orient only (no resize); augmentation on train only (crop 0-20%, rotation +-15 deg, grayscale 15%).
Labelling rules: reflections on the underside of the water surface are not labelled; distant or blurred items are labelled
when identifiable; partly visible items are labelled by their visible extent when identifiable.

## Results (test, 22 photos, 199 items)
| Class | Precision* | Recall* | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|---|
| goggles | 0.672 | 0.562 | 0.608 | 0.238 |
| cap | 0.860 | 0.683 | 0.793 | 0.537 |
| all | 0.789 | 0.638 | 0.701 | 0.388 |

\* at the operating point: confidence 0.35, NMS IoU 0.3, match IoU 0.5. `results/results_table.md` lists the
precision/recall that Ultralytics reports at its own F1-optimal confidence, so those two columns differ.
Counting per photograph: MAE 1.09, RMSE 1.46, both classes exact in 3/22 photos (`results/test_counts.jsonl`).

## Repository layout
```text
train_yolov8n.py       download the Roboflow export (optional), letterbox to 640, fine-tune yolov8n.pt
letterbox.py           aspect-preserving resize + gray padding, used in training and on the Jetson
tune_threshold.py      precision / recall / F1 for a grid of confidence and NMS IoU values
compare_test.py        ground truth vs prediction images for a split
collect_experiment.py  collects metrics, logs, plots and environment into one folder
models/                model_X3.pt (final weights; ONNX export in jetson/models/model_X3.onnx)
jetson/process/        runs on the Jetson Nano (TensorRT, no PyTorch): predict_folder.py, capture_predict.py
jetson/capture/        gopro_capture.py, runs on the Raspberry Pi (Open GoPro HTTP API, open_gopro for BLE)
jetson/computer/       shoot.py, runs on the computer: one command to capture, predict on the Jetson and copy results back
results/               metrics, training curves, evaluation plots, per-photo test counts
```

## Training (any computer)
```bash
pip install -r requirements.txt
# put the Roboflow version 4 YOLOv8 export in datasets/dataset-real-g3nyo-v4
# (or set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION in .env and omit --data)
python3 train_yolov8n.py --data datasets/dataset-real-g3nyo-v4 --epochs 100 --patience 100 --name model_X3
python3 tune_threshold.py --data datasets/dataset-real-g3nyo-v4 --split valid --model models/model_X3.pt --iou 0.3 0.4 0.5 0.7
python3 compare_test.py   --data datasets/dataset-real-g3nyo-v4 --split test  --model models/model_X3.pt --out runs/test_compare
```
Paper settings: imgsz 640, batch 8, seed 0, 100 epochs without early stopping, best checkpoint by validation fitness
(Ultralytics 8.4.173, PyTorch 2.14.1, Apple M4 / MPS).

## Jetson Nano (JetPack 4.6, TensorRT 8.2, Python 3.6)
Copy `jetson/` to the Jetson as `~/pool_detect/`. The first run builds `models/model_X3_fp16.engine` from the ONNX file with
`trtexec` (about 10 minutes, once).
```bash
cd ~/pool_detect
python3 process/predict_folder.py SESSION                        # predict images/SESSION/raw/
python3 process/capture_predict.py 10 --pi user@PI_IP            # Pi takes 10 photos with the GoPro, then predict
```
From the computer (photos, predictions, side-by-side comparisons and `results.jsonl` are copied back, nothing is kept
on the Pi or the Jetson):
```bash
python3 jetson/computer/shoot.py 10 --jetson user@JETSON_IP --pi user@PI_IP --session test1
```
Each line of `results.jsonl`: `file`, `counts {cap, glasses}`, `total_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms`.

## License
AGPL-3.0 (the training code and weights build on Ultralytics YOLO, which is AGPL-3.0).
