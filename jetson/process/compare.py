"""Side-by-side ORIGINAL vs PREDICTION images and contact sheets, same style as compare_test.py.

GoPro photos have no ground truth, so the left half is the photo itself (640 letterbox) instead.
"""
from pathlib import Path

import cv2
import numpy as np

from letterbox import letterbox

SIZE = 640
HEAD = 100  # header band with the counts
PER_SHEET, COLS = 12, 3


def _title(img, text):
    cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def pair(raw, predicted, counts, name="", confs=()):
    """raw / predicted: full-size BGR images (predicted already has boxes drawn).

    counts: {"cap": n, "glasses": m}; confs: [("cap", 0.82), ...] for the header line.
    """
    left = _title(letterbox(raw, SIZE)[0], "ORIGINAL")
    right = letterbox(predicted, SIZE)[0]
    if not any(counts.values()):
        cv2.putText(right, "no boxes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    _title(right, "PREDICTION")
    body = np.hstack([left, np.full((SIZE, 6, 3), 255, np.uint8), right])
    # header: what was found, readable even in the half-size sheets
    head = np.full((HEAD, body.shape[1], 3), 255, np.uint8)
    found = "   ".join("{}: {}".format(k, v) for k, v in counts.items())
    cv2.putText(head, "{}   total: {}".format(found, sum(counts.values())), (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    detail = ", ".join("{} {:.2f}".format(c, f) for c, f in confs) or "nothing detected"
    cv2.putText(head, "{}   [{}]".format(name, detail)[:110], (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    return np.vstack([head, body])


def sheets(compare_dir, out_dir):
    """Write sheet_N.jpg (12 half-size pairs, 3 per row) from every compare_*.jpg; returns the paths."""
    pairs = []
    for p in sorted(Path(compare_dir).glob("compare_*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            pairs.append(cv2.copyMakeBorder(small, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    for old in Path(out_dir).glob("sheet_*.jpg"):
        old.unlink()
    written = []
    for i in range(0, len(pairs), PER_SHEET):
        chunk = pairs[i:i + PER_SHEET]
        chunk += [np.full_like(pairs[0], 255)] * (-len(chunk) % COLS)
        grid = np.vstack([np.hstack(chunk[j:j + COLS]) for j in range(0, len(chunk), COLS)])
        path = Path(out_dir) / "sheet_{}.jpg".format(i // PER_SHEET + 1)
        cv2.imwrite(str(path), grid)
        written.append(path)
    return written
