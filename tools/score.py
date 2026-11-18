"""Score outputs/ against GT instance PNGs using the Day 2 rules (slide 58) + F1@0.95 (spec).

P1 = 7.5 * mean over images of [sum IoU of 1:1 matches / max(#gt, #pred)]   (class-agnostic, Hungarian)
P2 = 7.5 * mean over images of [#matched (IoU>0) / max(#gt, #pred)]
Latency (10): <=1.00 s -> 10, <=2.00 -> 7, <=3.00 -> 4, else 0; scaled by P/7.5 if P < 7.5.
Usage: python tools/score.py --pred outputs --gt-dir gt_val --latency 0.85
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt, pr):
    g_ids = np.unique(gt)[1:]
    p_ids = np.unique(pr)[1:]
    if len(g_ids) == 0 or len(p_ids) == 0:
        return np.zeros((len(g_ids), len(p_ids))), g_ids, p_ids
    # joint histogram of (gt id, pred id)
    gmax, pmax = int(gt.max()) + 1, int(pr.max()) + 1
    joint = np.bincount(gt.astype(np.int64).ravel() * pmax + pr.astype(np.int64).ravel(),
                        minlength=gmax * pmax).reshape(gmax, pmax)
    ga, pa = joint.sum(1), joint.sum(0)
    inter = joint[np.ix_(g_ids, p_ids)].astype(float)
    union = ga[g_ids][:, None] + pa[p_ids][None, :] - inter
    return inter / np.maximum(union, 1), g_ids, p_ids


def latency_pts(s):
    return 10 if s <= 1.0 else 7 if s <= 2.0 else 4 if s <= 3.0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--latency", type=float, default=None, help="median s/image on the Pi")
    a = ap.parse_args()
    preds = {d["image_name"]: d for d in json.load(open(os.path.join(a.pred, "predictions.json"), encoding="utf-8"))}
    ious, cnts, f1s, rows = [], [], [], []
    tp = fp = fn = 0
    for gpath in sorted(glob.glob(os.path.join(a.gt_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(gpath))[0]
        gt = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)
        pp = os.path.join(a.pred, "seg_mask", stem + ".png")
        pr = cv2.imread(pp, cv2.IMREAD_UNCHANGED) if os.path.exists(pp) else np.zeros_like(gt)
        assert pr.shape == gt.shape, f"{stem}: shape {pr.shape} vs gt {gt.shape}"
        M, g_ids, p_ids = iou_matrix(gt, pr)
        ng, np_ = len(g_ids), len(p_ids)
        denom = max(ng, np_, 1)
        if M.size:
            r, c = linear_sum_assignment(-M)
            v = M[r, c]
        else:
            v = np.array([])
        ious.append(v.sum() / denom)
        cnts.append((v > 0).sum() / denom)
        t = int((v >= 0.95).sum())
        tp, fp, fn = tp + t, fp + np_ - t, fn + ng - t
        rows.append((stem, ng, np_, v.mean() if len(v) else 0, ious[-1], cnts[-1], t))
    print(f"{'image':28s} gt  pred  mIoU(matched)  IoU-score  count-score  TP@.95")
    for s, ng, np_, mi, i, c, t in rows:
        print(f"{s:28s} {ng:3d} {np_:4d}   {mi:.3f}         {i:.3f}      {c:.3f}       {t}")
    P1, P2 = 7.5 * np.mean(ious), 7.5 * np.mean(cnts)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"P1 (IoU) = {P1:.2f}/7.5   P2 (count) = {P2:.2f}/7.5   P = {P1 + P2:.2f}/15   F1@0.95 = {f1:.3f}")
    if a.latency is not None:
        L = latency_pts(a.latency) * (min(P1 + P2, 7.5) / 7.5)
        print(f"latency {a.latency:.2f}s -> {L:.1f}/10   TOTAL = {P1 + P2 + L:.2f}/25")


if __name__ == "__main__":
    main()
