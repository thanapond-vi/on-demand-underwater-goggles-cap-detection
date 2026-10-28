"""YOLOv8n detector on the Jetson Nano GPU with TensorRT; needs no torch / ultralytics.

Pre/post-processing matches training and predict_yolov8n.py: 640x640 letterbox (letterbox.py),
class-aware NMS. Checked against ultralytics best.pt on the 50 test images (identical detections).
"""
import ctypes
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

from letterbox import boxes_to_original, letterbox

NAMES = ["cap", "glasses"]
COLORS = [(0, 200, 0), (255, 140, 0)]



def _load_cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so.10.2"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise OSError("libcudart not found; add /usr/local/cuda/lib64 to LD_LIBRARY_PATH")


_cudart = _load_cudart()
H2D, D2H = 1, 2


def _check(err, what):
    if err != 0:
        raise RuntimeError("{} failed with CUDA error {}".format(what, err))


def ensure_engine(engine_path, onnx_path=None):
    """Build the TensorRT engine from the ONNX next to it the first time (5-15 min on a Nano)."""
    engine_path = Path(engine_path)
    if engine_path.exists():
        return engine_path
    onnx_path = Path(onnx_path or engine_path.parent / (engine_path.stem.replace("_fp16", "") + ".onnx"))
    if not onnx_path.exists():
        raise SystemExit("no engine {} and no {} to build it from".format(engine_path, onnx_path))
    print("first run: building TensorRT engine from {} (5-15 min, only once)...".format(onnx_path.name), flush=True)
    log = engine_path.with_suffix(".build.log")
    with open(str(log), "w") as fh:
        r = subprocess.run(["/usr/src/tensorrt/bin/trtexec", "--onnx=" + str(onnx_path),
                            "--saveEngine=" + str(engine_path), "--fp16", "--workspace=1024"],
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not engine_path.exists():
        raise SystemExit("engine build failed, see {}".format(log))
    print("engine ready: {}".format(engine_path), flush=True)
    return engine_path


class TrtModel:
    """Fixed-shape TensorRT engine: one input (1,3,S,S) float32, one output (1,4+nc,N) float32."""

    def __init__(self, engine_path):
        engine_path = ensure_engine(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("could not load engine {}".format(engine_path))
        self.ctx = self.engine.create_execution_context()
        e = self.engine
        self.tensor_api = hasattr(e, "num_io_tensors")  # TensorRT >= 8.5 (JetPack 5/6); 8.2 = JetPack 4.6
        if self.tensor_api:
            names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            specs = [(n, e.get_tensor_shape(n), e.get_tensor_dtype(n),
                      e.get_tensor_mode(n) == trt.TensorIOMode.INPUT) for n in names]
        else:
            specs = [(e.get_binding_name(i), e.get_binding_shape(i), e.get_binding_dtype(i),
                      e.binding_is_input(i)) for i in range(e.num_bindings)]
        self.host, self.dev, self.bindings = [], [], []
        for name, shape, dtype, _ in specs:
            arr = np.empty(tuple(shape), dtype=trt.nptype(dtype))
            ptr = ctypes.c_void_p()
            _check(_cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(arr.nbytes)), "cudaMalloc")
            if self.tensor_api:
                self.ctx.set_tensor_address(name, ptr.value)
            self.host.append(arr)
            self.dev.append(ptr)
            self.bindings.append(ptr.value)
        self.inp = [i for i, sp in enumerate(specs) if sp[3]][0]
        self.out = [i for i, sp in enumerate(specs) if not sp[3]][0]
        self.imgsz = self.host[self.inp].shape[-1]
        self(np.zeros(self.host[self.inp].shape, np.float32))  # warm-up: first run is always slow

    def __call__(self, blob):
        src = np.ascontiguousarray(blob, dtype=self.host[self.inp].dtype)
        _check(_cudart.cudaMemcpy(self.dev[self.inp], src.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_size_t(src.nbytes), H2D), "cudaMemcpy H2D")
        ok = self.ctx.execute_async_v3(0) if self.tensor_api else self.ctx.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        out = self.host[self.out]
        _check(_cudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), self.dev[self.out],
                                  ctypes.c_size_t(out.nbytes), D2H), "cudaMemcpy D2H")
        return out

    def close(self):
        for p in self.dev:
            _cudart.cudaFree(p)
        self.dev = []


def nms(boxes, scores, iou):
    """Greedy NMS on xyxy boxes; returns kept indices sorted by score."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=int)


def postprocess(out, conf, iou, max_det=300):
    """YOLOv8 head (1, 4+nc, N) -> xyxy (letterbox pixels), cls, conf with class-aware NMS."""
    pred = out[0].T  # (N, 4+nc)
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    cf = scores[np.arange(len(cls)), cls]
    m = cf > conf
    pred, cls, cf = pred[m], cls[m], cf[m]
    if not len(cf):
        return np.zeros((0, 4)), cls, cf
    xy, wh = pred[:, :2], pred[:, 2:4] / 2
    xyxy = np.concatenate([xy - wh, xy + wh], 1)
    keep = nms(xyxy + cls[:, None] * 7680.0, cf, iou)[:max_det]  # offset per class = class-aware NMS
    return xyxy[keep], cls[keep], cf[keep]


def detect(model, img, conf, iou):
    """Return (xyxy in original pixels, cls, conf, speed) with speed in ms like ultralytics r.speed."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    boxed, scale, px, py = letterbox(img, model.imgsz)  # explicit resize + pad, same as training
    blob = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    t1 = time.perf_counter()
    out = model(blob)
    t2 = time.perf_counter()
    xyxy, cls, cf = postprocess(out, conf, iou)
    xyxy = boxes_to_original(xyxy, scale, px, py, w, h)
    t3 = time.perf_counter()
    speed = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000}
    return xyxy, cls, cf, speed


def draw(img, xyxy, cls, conf):
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 300))
    for (x1, y1, x2, y2), c, cf in zip(xyxy.astype(int), cls, conf):
        color = COLORS[c % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, "{} {:.2f}".format(NAMES[c], cf), (x1, max(thick * 10, y1 - thick * 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, thick / 3, color, thick)
    return img
