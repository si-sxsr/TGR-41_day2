"""Auto-label beans on white paper: background-corrected threshold at full resolution.

Output: labels/<class>/<stem>.png  (uint16, 0 = background, 1..N = bean instance)
        labels/review/<class>_<stem>.jpg  (overlay for checking by eye)
Usage:  python tools/autolabel.py --src dataset --out labels
"""
import argparse
import glob
import os

import cv2
import numpy as np


def bean_score(img_bgr, scale):
    """Per-pixel 'bean-ness' = darkness vs local background + colour difference vs background."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    small = cv2.resize(lab, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    # beans are darker than paper -> grey closing (max filter) removes them from L
    bg_l = cv2.GaussianBlur(cv2.morphologyEx(small[..., 0], cv2.MORPH_CLOSE, k), (0, 0), 15)
    bg_a = cv2.GaussianBlur(cv2.medianBlur(small[..., 1].astype(np.uint8), 31), (0, 0), 15).astype(np.float32)
    bg_b = cv2.GaussianBlur(cv2.medianBlur(small[..., 2].astype(np.uint8), 31), (0, 0), 15).astype(np.float32)
    h, w = lab.shape[:2]
    up = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    dark = up(bg_l) - lab[..., 0]
    chroma = np.sqrt((lab[..., 1] - up(bg_a)) ** 2 + (lab[..., 2] - up(bg_b)) ** 2)
    score = np.clip(dark, 0, None) + 1.5 * chroma
    return cv2.GaussianBlur(score, (5, 5), 0), dark


def fill_holes(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out


def label_image(img):
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    score, dark = bean_score(img, scale)
    small = cv2.resize(score, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s8 = np.clip(small * 255 / max(small.max(), 1e-6), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t * small.max() / 255, 12.0)
    m = (score > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.zeros((h, w), np.uint16)
    big = areas[areas > 0.002 * h * w]
    med = np.median(big) if len(big) else np.median(areas)
    out = np.zeros((h, w), np.uint16)
    nid = 0
    margin = 5
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < 0.25 * med or a > 4 * med:
            continue
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue  # paper edge / table at border
        comp = (cc[y:y + bh, x:x + bw] == i).astype(np.uint8)
        if dark[y:y + bh, x:x + bw][comp > 0].mean() < 8:
            continue  # coloured but not darker than paper (highlighter marks)
        comp = fill_holes(comp)
        nid += 1
        out[y:y + bh, x:x + bw][comp > 0] = nid
    return out


def overlay(img, inst):
    vis = img.copy()
    rng = np.random.default_rng(0)
    for i in range(1, int(inst.max()) + 1):
        m = (inst == i).astype(np.uint8)
        c = rng.integers(60, 255, 3).tolist()
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, c, 6)
        x, y, _, _ = cv2.boundingRect(cnts[0])
        cv2.putText(vis, str(i), (x, y - 8), 0, 2.5, (0, 0, 255), 5)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out", default="labels")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "review"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.src, "*", "*.jpg"))):
        cls = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)  # applies EXIF orientation
        inst = label_image(img)
        os.makedirs(os.path.join(a.out, cls), exist_ok=True)
        cv2.imwrite(os.path.join(a.out, cls, stem + ".png"), inst)
        vis = overlay(img, inst)
        s = 1200 / max(vis.shape[:2])
        cv2.imwrite(os.path.join(a.out, "review", f"{cls}_{stem}.jpg"), cv2.resize(vis, None, fx=s, fy=s))
        areas = np.bincount(inst.ravel())[1:]
        print(f"{cls:10s} {stem:32s} n={inst.max():3d}  area med={int(np.median(areas)) if len(areas) else 0}")


if __name__ == "__main__":
    main()
