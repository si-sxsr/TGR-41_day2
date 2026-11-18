"""Train YOLO-seg models (CPU) on yolo_data/beans.yaml.
Usage: python train.py --models yolov8n-seg.pt yolo11n-seg.pt yolo26n-seg.pt --epochs 40
Labels come from tools/autolabel.py (threshold on white paper) -> tools/make_yolo.py."""
import argparse
import time

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["yolov8n-seg.pt", "yolo11n-seg.pt", "yolo26n-seg.pt"])
ap.add_argument("--data", default="yolo_data/beans.yaml")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

for m in a.models:
    t = time.time()
    name = m.replace(".pt", "")
    YOLO(m).train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=8, device="cpu", workers=2,
                  cache="ram", project="runs", name=name, exist_ok=True, plots=False,
                  copy_paste=0.3, degrees=180, flipud=0.5, mosaic=1.0, close_mosaic=5,
                  patience=100, seed=0, deterministic=True)
    print(f"### {name} done in {(time.time() - t) / 60:.1f} min", flush=True)
