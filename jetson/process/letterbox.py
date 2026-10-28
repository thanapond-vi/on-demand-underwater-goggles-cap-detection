"""Shared 640x640 letterbox resize used by both training and inference.

Scale the longer side to `size` (aspect ratio preserved), then pad the shorter
side equally on both edges with gray (114) to a `size` x `size` square.
"""
import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


def letterbox(img, size=640):
    """Return (padded_img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, pad_x, pad_y


def labels_to_letterbox(rows, w, h, scale, pad_x, pad_y, size=640):
    """Map YOLO rows (cls, xc, yc, bw, bh normalized to the source image) into the letterboxed image."""
    out = []
    for c, xc, yc, bw, bh in rows:
        out.append((int(c),
                    (xc * w * scale + pad_x) / size,
                    (yc * h * scale + pad_y) / size,
                    bw * w * scale / size,
                    bh * h * scale / size))
    return out


def boxes_to_original(xyxy, scale, pad_x, pad_y, w, h):
    """Map xyxy boxes predicted on the letterboxed image back to source-image pixels."""
    b = xyxy.copy().astype(float)
    b[:, [0, 2]] = ((b[:, [0, 2]] - pad_x) / scale).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - pad_y) / scale).clip(0, h)
    return b
