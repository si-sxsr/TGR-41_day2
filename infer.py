"""Day 2 inference: instance segmentation of coffee beans.

python infer.py --input test/ --weights weights/ --out outputs/

--weights is a folder. Loading priority: *_ncnn_model/ folder > *.onnx > *.pt (first match, sorted).
Writes outputs/seg_mask/<stem>.png (uint16, 0 = bg, 1..N), outputs/predictions.json, outputs/vis/<stem>.jpg.
CPU only, no GUI, no downloads, no TTA.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

CLASSES = ["black", "broken", "good", "husk", "immature", "parchment", "shell"]
EXTS = (".jpg", ".jpeg", ".png")
WORK_LONG = 1024  # decode at the smallest JPEG scale (1/4, 1/2) whose long side is still >= this; model input is 448


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_weights(folder):
    """Candidates in priority order: *_ncnn_model/ > *.onnx > *.pt."""
    if not os.path.isdir(folder):
        die(f"weights folder not found: {folder}")
    cands = []
    for pat in ("*_ncnn_model", "*.onnx", "*.pt"):
        cands += sorted(glob.glob(os.path.join(folder, pat)))
    if not cands:
        die(f"no *_ncnn_model / *.onnx / *.pt in {folder}")
    return cands


def load_model(folder, imgsz):
    """Load the first candidate that works (e.g. falls back to .onnx/.pt if ncnn is not installed)."""
    from ultralytics import YOLO
    for w in find_weights(folder):
        try:
            m = YOLO(w, task="segment")
            m.predict(np.zeros((480, 640, 3), np.uint8), imgsz=imgsz, verbose=False, device="cpu")  # warm-up
            return m, w
        except Exception as e:
            print(f"WARNING: cannot load {w}: {e}", file=sys.stderr)
    die("no loadable model in weights folder")


def full_size(path):
    """Image size after EXIF orientation, read from the header only."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
    return h, w


def read_work(path, h, w):
    """Decode at reduced size (fast on Pi); cv2 applies EXIF orientation."""
    flag = cv2.IMREAD_COLOR
    for f, r in ((cv2.IMREAD_REDUCED_COLOR_4, 4), (cv2.IMREAD_REDUCED_COLOR_2, 2)):
        if max(h, w) / r >= WORK_LONG:
            flag = f
            break
    img = cv2.imread(path, flag)
    if img is None:
        die(f"cannot read image: {path}")
    return img


def predict_instances(model, img, imgsz, conf, retina):
    """Return list of (cls_id, score, bool mask) sorted by score desc (mask resized later if needed)."""
    r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=retina, verbose=False, device="cpu")[0]
    if r.masks is None or len(r.boxes) == 0:
        return []
    masks = r.masks.data.cpu().numpy() > 0.5
    h0, w0 = img.shape[:2]
    mh, mw = masks.shape[1:]
    if (mh, mw) != (h0, w0):  # letterboxed input size: remove padding before resizing to work size
        g = min(mh / h0, mw / w0)
        px, py = (mw - w0 * g) / 2, (mh - h0 * g) / 2
        y0, x0 = int(round(py - 0.1)), int(round(px - 0.1))
        y1, x1 = int(round(mh - py + 0.1)), int(round(mw - px + 0.1))
        masks = masks[:, y0:y1, x0:x1]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    sc = r.boxes.conf.cpu().numpy()
    order = np.argsort(-sc)
    return [(cls[i], float(sc[i]), masks[i]) for i in order]


def build_instance_map(insts, wh, min_frac=1e-5):
    """Paint highest-score first at the mask resolution (no extra resize: finalize() upsamples each
    instance to full size); each pixel belongs to one instance."""
    hw, ww = insts[0][2].shape if insts else wh
    small = np.zeros((hw, ww), np.uint16)
    min_area = max(3, int(min_frac * hw * ww))
    kept = []
    for c, s, m in insts:
        free = m & (small == 0)
        if free.sum() < min_area:
            continue
        kept.append((c, s))
        small[free] = len(kept)
    return small, kept


def refine(small, img):
    """Snap each YOLO mask to the real bean edge on the white paper (pixel-accurate at work size).
    Paper colour is estimated from pixels outside the predicted beans (inpainted low-res map),
    bean pixels = darker or more coloured than paper (Otsu threshold near the beans).
    Each instance may only grow into a thin ring around its own YOLO mask and never into a neighbour."""
    from scipy.ndimage import find_objects
    hw, ww = img.shape[:2]
    lab_map = cv2.resize(small, (ww, hw), interpolation=cv2.INTER_NEAREST)
    if lab_map.max() == 0:
        return lab_map
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = 160 / max(hw, ww)
    lo = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    beans_lo = cv2.resize((lab_map > 0).astype(np.uint8), (lo.shape[1], lo.shape[0]), interpolation=cv2.INTER_NEAREST)
    beans_lo = cv2.dilate(beans_lo, np.ones((5, 5), np.uint8))
    bg = cv2.cvtColor(cv2.inpaint(lo, beans_lo, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2LAB)
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 3), (ww, hw), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    score = np.clip(bg[..., 0] - lab[..., 0], 0, None) + 1.5 * np.sqrt(
        (lab[..., 1] - bg[..., 1]) ** 2 + (lab[..., 2] - bg[..., 2]) ** 2)
    score = cv2.GaussianBlur(score, (3, 3), 0)
    near = cv2.dilate((lab_map > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    v = np.clip(score[near], 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = score > max(t, 10)
    out = np.zeros_like(lab_map)
    for idx, sl in enumerate(find_objects(lab_map)):
        if sl is None:
            continue
        i = idx + 1
        m = lab_map == i  # full-frame bool is fine at work size; use crop for speed
        ys, xs = sl
        area = int(m[sl].sum())
        r = max(2, int(0.08 * np.sqrt(area)))
        y0, y1 = max(ys.start - r, 0), min(ys.stop + r, hw)
        x0, x1 = max(xs.start - r, 0), min(xs.stop + r, ww)
        mc = m[y0:y1, x0:x1].astype(np.uint8)
        ring = cv2.dilate(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        others = (lab_map[y0:y1, x0:x1] > 0) & ~mc.astype(bool)
        new = ring & fg[y0:y1, x0:x1] & ~others & (out[y0:y1, x0:x1] == 0)
        new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        if new.sum() < 0.6 * area or new.sum() > 1.6 * area:
            new = mc.astype(bool) & (out[y0:y1, x0:x1] == 0)  # refinement unreliable -> keep YOLO mask
        out[y0:y1, x0:x1][new] = i
    return out


def finalize(small, kept, names, full_hw):
    """Upscale each instance on its own crop (bilinear -> smooth edges), keep the largest part,
    fill holes, and paint into the full-size map so that mask == polygon interior.
    Works per bounding box, so cost does not grow with full image size x number of beans."""
    from scipy.ndimage import find_objects
    H, W = full_hw
    hs, ws = small.shape
    sy, sx = H / hs, W / ws
    out = np.zeros((H, W), np.uint16)
    objs = []
    n = 0
    for idx, sl in enumerate(find_objects(small)):
        if sl is None:
            continue
        i = idx + 1
        ys0, ys1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, hs)
        xs0, xs1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, ws)
        y0, y1 = int(round(ys0 * sy)), min(int(round(ys1 * sy)), H)
        x0, x1 = int(round(xs0 * sx)), min(int(round(xs1 * sx)), W)
        m = (small[ys0:ys1, xs0:xs1] == i).astype(np.float32)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
        region = out[y0:y1, x0:x1]
        m &= region == 0  # never steal pixels of an earlier instance
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 3 or cv2.contourArea(c) < 4:
            continue
        keep = np.zeros(m.shape, np.uint8)
        cv2.drawContours(keep, [c], -1, 1, thickness=cv2.FILLED)
        keep = (keep > 0) & (region == 0)
        n += 1
        region[keep] = n
        pts = c.reshape(-1, 2) + [x0, y0]
        objs.append({"id": n, "class": str(names[int(kept[i - 1][0])]).lower(), "polygon": pts.astype(int).tolist()})
    return out, objs


def draw_vis(img_small, full_hw, objs, path):
    """Overlay drawn from the polygons on the work-size image (cheap)."""
    h, w = img_small.shape[:2]
    k = w / full_hw[1]
    vis = img_small.copy()
    over = vis.copy()
    rng = np.random.default_rng(1)
    for o in objs:
        pts = (np.asarray(o["polygon"]) * k).astype(np.int32)
        cv2.fillPoly(over, [pts], rng.integers(50, 255, 3).tolist())
    vis = cv2.addWeighted(over, 0.5, vis, 0.5, 0)
    for o in objs:
        pts = (np.asarray(o["polygon"]) * k).astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 1)
        x, y = pts.min(0)
        cv2.putText(vis, f'{o["id"]}:{o["class"]}', (int(x), max(int(y) - 3, 10)), 0, 0.4, (0, 0, 255), 1)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--retina", action="store_true", help="upsample protos to work size before threshold")
    ap.add_argument("--refine", action="store_true", help="snap mask edges to the bean/paper boundary")
    ap.add_argument("--no-vis", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.input):
        die(f"input folder not found: {a.input}")
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(a.input, f)))
    if not files:
        die(f"no .jpg/.jpeg/.png images in {a.input}")
    stems = [os.path.splitext(f)[0].lower() for f in files]
    if len(set(stems)) != len(stems):
        die("duplicate image stems in input (PNG outputs would overwrite each other)")
    if os.path.abspath(a.out) == os.path.abspath(a.input):
        die("--out must differ from --input")

    model, wpath = load_model(a.weights, a.imgsz)
    bad = [n for n in model.names.values() if str(n).lower() not in CLASSES]
    if bad:
        print(f"WARNING: model classes not in spec: {bad[:5]}", file=sys.stderr)

    os.makedirs(os.path.join(a.out, "seg_mask"), exist_ok=True)
    if not a.no_vis:
        os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    results, times = [], []
    for f in files:
        t0 = time.perf_counter()
        p = os.path.join(a.input, f)
        try:
            H, W = full_size(p)
        except Exception as e:
            die(f"cannot read image {p}: {e}")
        img = read_work(p, H, W)
        insts = predict_instances(model, img, a.imgsz, a.conf, a.retina)
        small, kept = build_instance_map(insts, img.shape[:2])
        if a.refine:
            small = refine(small, img)
        full, objs = finalize(small, kept, model.names, (H, W))
        stem = os.path.splitext(f)[0]
        if full.max() <= 255:
            full = full.astype(np.uint8)  # same integer ids, half the bytes -> much faster PNG write
        if not cv2.imwrite(os.path.join(a.out, "seg_mask", stem + ".png"), full, [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]):
            die(f"failed to write mask for {f}")
        results.append({"image_name": f, "objects": objs})
        dt = time.perf_counter() - t0
        times.append(dt)
        if not a.no_vis:
            draw_vis(img, (H, W), objs, os.path.join(a.out, "vis", stem + ".jpg"))
        print(f"{f}: {len(objs)} beans, {dt * 1000:.0f} ms")

    with open(os.path.join(a.out, "predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False)
    print(f"done {len(files)} images | median {np.median(times) * 1000:.0f} ms/image | model {wpath}")


if __name__ == "__main__":
    main()
