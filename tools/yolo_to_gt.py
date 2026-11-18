"""YOLO-seg polygons -> instance PNG GT (for tools/score.py). Usage: python tools/yolo_to_gt.py <split_dir> <out_img_dir> <out_gt_dir>"""
import glob, os, shutil, sys
import cv2, numpy as np
src, oimg, ogt = sys.argv[1:4]
os.makedirs(oimg, exist_ok=True); os.makedirs(ogt, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "images", "*.jpg"))):
    stem = os.path.splitext(os.path.basename(f))[0]
    h, w = cv2.imread(f).shape[:2]
    m = np.zeros((h, w), np.uint16)
    for i, ln in enumerate(open(os.path.join(src, "labels", stem + ".txt")).read().split("\n")):
        v = ln.split()
        if len(v) < 7: continue
        p = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        cv2.fillPoly(m, [p], i + 1)
    shutil.copy(f, os.path.join(oimg, stem + ".jpg"))
    cv2.imwrite(os.path.join(ogt, stem + ".png"), m)
    print(stem[:40], len(np.unique(m)) - 1)
