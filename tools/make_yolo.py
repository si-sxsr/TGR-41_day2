"""Build YOLO-seg dataset from dataset_official + labels/. train = circle/grid, val = square.
Images are resized to long side 1280 (EXIF already applied by cv2.imread) to speed up loading."""
import glob
import os

import cv2
import numpy as np

CLASSES = ["black", "broken", "good", "husk", "immature", "parchment", "shell"]
SRC, LBL, OUT, LONG = "dataset_official", "labels", "yolo_data", 1280

for f in sorted(glob.glob(f"{SRC}/*/*.jpg")):
    cls = os.path.basename(os.path.dirname(f))
    stem = os.path.splitext(os.path.basename(f))[0]
    split = "val" if stem.startswith("square_") else "train"
    img = cv2.imread(f)
    inst = cv2.imread(f"{LBL}/{cls}/{stem}.png", cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    s = LONG / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    cv2.imwrite(f"{OUT}/images/{split}/{stem}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for i in range(1, int(inst.max()) + 1):
        m = (inst == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float64)
        c[:, 0] /= w
        c[:, 1] /= h
        lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in c.ravel()))
    open(f"{OUT}/labels/{split}/{stem}.txt", "w").write("\n".join(lines) + "\n")
    print(split, stem, len(lines))

with open(f"{OUT}/beans.yaml", "w") as fh:
    fh.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\nnames:\n")
    for i, c in enumerate(CLASSES):
        fh.write(f"  {i}: {c}\n")
