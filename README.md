# Day 2 — Coffee bean instance segmentation (YOLO11n-seg)

## Run

```sh
python infer.py --input test/ --weights weights/ --out outputs/
```

Outputs: `outputs/seg_mask/<stem>.png`, `outputs/predictions.json`, `outputs/vis/<stem>.jpg`.
Exit code 0 on success; nonzero (with an error message) if input/weights are missing, there are no images,
an image cannot be read, or two inputs share a stem.

## Install (Raspberry Pi 4/5, 64-bit OS, CPU only)

```sh
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Weights

`weights/` holds the same trained YOLO11n-seg model in several formats. `infer.py` tries them in this order
and uses the first one that loads: `*_ncnn_model/` (fastest on ARM) → `*.onnx` → `*.pt`.
Nothing is downloaded at inference time.

## Pipeline (infer.py)

1. Images sorted by name; `.jpg/.jpeg/.png` (any case). Full size is read from the header after EXIF orientation.
2. The JPEG is decoded at 1/4 (or 1/2) size with `cv2.IMREAD_REDUCED_COLOR_*` (long side ≥ 1024) to save time on the Pi.
3. YOLO11n-seg at imgsz 448, conf 0.25 (no TTA, no random augmentation).
4. Instances painted highest-score first, so every pixel belongs to at most one bean.
5. Each instance is upscaled on its own crop to the original resolution (bilinear, threshold 0.5),
   only its largest connected part is kept and holes are filled, so the PNG mask equals the polygon interior.
6. `seg_mask/*.png`: single-channel integer image, same size as the EXIF-corrected original, 0 = background,
   1..N = bean id. Saved as uint8 when N ≤ 255 (same integer values), otherwise uint16.
7. `predictions.json`: polygon = `cv2.findContours(..., CHAIN_APPROX_NONE)` of the exact painted mask,
   `id` matches the PNG value, class names in lowercase.

## Training (train.py)

- Data: our own pile photos (beans touching, like the test set) on white paper, labelled by hand as
  polygons in Roboflow (`dataset_rf/`, export "YOLOv8", images 432x432): 27 train / 7 valid, classes good + shell.
- Start weights: YOLO11n-seg (COCO) first fine-tuned briefly on the organizer images
  (`tools/autolabel.py` threshold labels), then fine-tuned on the pile set.
- CPU training, imgsz 448, batch 8, mosaic, copy-paste 0.3, rotation 180°, flips.

```sh
python train.py --models yolo11n-seg-beanstart.pt --data dataset_rf/beans_rf.yaml --imgsz 448 --epochs 80
```

## Self-check

`tools/score.py` scores an output folder with the Day 2 rules (class-agnostic 1:1 matching, mean IoU,
counting, F1@0.95) against GT instance PNGs.
