# TGR-41 · Day 2 · Coffee Bean Instance Segmentation

หา **ทุกเม็ด** ในภาพกองเมล็ดกาแฟ แล้วส่ง mask รายเม็ด (`seg_mask/`) + polygon และคลาสต่อเม็ด (`predictions.json`)
โมเดล: **YOLO11n-seg** (2.9 M params) รันบน CPU ของ Raspberry Pi ผ่าน NCNN

## รัน inference (คำสั่งมาตรฐาน)

```sh
python infer.py --input <test_folder/> --weights weights/ --out outputs/
```

- `--input`: โฟลเดอร์ภาพ รองรับ `.jpg` `.jpeg` `.png` (ไม่สนตัวพิมพ์เล็ก/ใหญ่) เรียงตามชื่อไฟล์ก่อนประมวลผล
- `--weights`: โฟลเดอร์ (ไม่ต้องระบุชื่อไฟล์) `infer.py` เลือกโมเดลเอง ดูหัวข้อ "ไฟล์ใน weights/"
- `--out`: โฟลเดอร์ผลลัพธ์ (ต้องไม่ใช่โฟลเดอร์เดียวกับ input) ไม่แก้ ย้าย หรือลบไฟล์ input
- ไม่มี GUI ไม่ดาวน์โหลดอะไรตอนรัน ไม่มี TTA / random augmentation ใช้ CPU อย่างเดียว
- ตัวเลือกเสริม (ไม่ต้องใช้ตอนตรวจ): `--conf 0.25` `--imgsz 448` `--no-vis`
- error → แสดงข้อความแล้วจบด้วย **exit code 1** ได้แก่ ไม่พบ input/weights, ไม่มีภาพ, อ่านภาพไม่ได้,
  ชื่อไฟล์ (stem) ซ้ำกันจน PNG จะเขียนทับกัน, เขียนผลไม่ได้ ไม่มีการข้ามภาพแบบเงียบ ๆ
- สำเร็จ → exit code 0 และพิมพ์เวลาต่อภาพ + median

## ติดตั้ง (สำหรับ inference)

Raspberry Pi 4/5, Raspberry Pi OS **64-bit**, Python 3.9–3.12

```sh
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # torch แบบ CPU (ultralytics ต้องใช้)
pip install -r requirements.txt
```

`requirements.txt` มีเฉพาะที่ `infer.py` ใช้: `ultralytics`, `torch`, `torchvision`, `ncnn`, `onnxruntime`,
`opencv-python-headless`, `numpy`, `scipy`, `pillow`

## ผลลัพธ์

```
outputs/
├── seg_mask/<ชื่อภาพ>.png   # 1 ไฟล์ต่อภาพ test เช่น img_01.jpg -> img_01.png
├── predictions.json
└── vis/<ชื่อภาพ>.jpg        # ภาพ overlay mask ไว้ดูด้วยตา (ไม่ใช่ mask จริง)
```

**`seg_mask/*.png`**
- ภาพ integer **channel เดียว** ขนาดเท่าภาพต้นฉบับ**หลังหมุนตาม EXIF orientation**
- `0` = พื้นหลัง, `1..N` = เม็ดที่ 1..N แต่ละพิกเซลเป็นของได้เม็ดเดียว
- บันทึกเป็น uint8 เมื่อ N ≤ 255 (ค่า id เหมือนกันทุกประการ แต่เขียนเร็วกว่า) และเป็น uint16 เมื่อ N > 255
  อ่านกลับด้วย `cv2.imread(path, cv2.IMREAD_UNCHANGED)` หรือ `np.array(Image.open(path))`
- ภาพที่ไม่พบเม็ด: PNG เป็น 0 ทั้งภาพ และ `objects` เป็น `[]`

**`predictions.json`** (UTF-8, 1 รายการต่อภาพ input ตามลำดับชื่อไฟล์)

```json
[
  {"image_name": "img_01.jpg",
   "objects": [
     {"id": 1, "class": "good", "polygon": [[412, 230], [413, 230], [414, 231], ...]},
     {"id": 2, "class": "shell", "polygon": [[518, 301], ...]}
   ]},
  {"image_name": "img_02.jpg", "objects": []}
]
```

- `id` ตรงกับค่าของเม็ดนั้นใน PNG ของภาพเดียวกัน (เริ่ม 1 ใหม่ทุกภาพ)
- `class` เป็นตัวพิมพ์เล็ก หนึ่งใน black, broken, good, husk, immature, parchment, shell
- `polygon` = `cv2.findContours(..., CHAIN_APPROX_NONE)` ของ mask ที่เขียนลง PNG จริง (เก็บจุดขอบครบ ไม่ลดจุด)
  หน่วยพิกเซลของภาพต้นฉบับ x = คอลัมน์ y = แถว อยู่ในขอบภาพเสมอ อย่างน้อย 3 จุด
- **mask กับ polygon ตรงกัน**: ทุกเม็ดเก็บเฉพาะชิ้นที่ใหญ่ที่สุดและเติมรูข้างใน ก่อนเขียนทั้ง PNG และ polygon
  จึงไม่มีส่วนของ mask ที่ polygon แทนไม่ได้

## วิธีการ (infer.py)

1. **อ่านขนาดจริง** จาก header ของไฟล์ (PIL) แล้วสลับกว้าง/สูงตาม EXIF orientation
2. **Decode ภาพแบบย่อ** ด้วย `cv2.IMREAD_REDUCED_COLOR_2/4` (ด้านยาว ≥ 1024) cv2 หมุนตาม EXIF ให้เอง
   ภาพ 12 MP ไม่ต้อง decode เต็มขนาด ประหยัดเวลาบน Pi
3. **YOLO11n-seg** imgsz 448 (letterbox) conf 0.25 → กรอบ + คลาส + mask ต่อเม็ด
4. **ไม่ให้เม็ดทับกัน**: ระบายเม็ดที่ความมั่นใจสูงก่อน เม็ดถัดไปได้เฉพาะพิกเซลที่ยังว่าง
5. **ขยาย mask กลับขนาดภาพต้นฉบับทีละเม็ด** เฉพาะในกรอบของเม็ดนั้น (bilinear แล้ว threshold 0.5 ขอบเรียบกว่า nearest)
   เวลาจึงไม่เพิ่มตามขนาดภาพ × จำนวนเม็ด
6. เขียน PNG (zlib level 1 + RLE) และ `vis/` วาดจาก polygon บนภาพย่อ

## ไฟล์ใน weights/

ทั้งหมดคือโมเดลเดียวกัน `infer.py` ลองโหลดตามลำดับนี้และใช้ตัวแรกที่โหลดได้

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `best_ncnn_model/` | ตัวหลัก NCNN เร็วที่สุดบน ARM CPU (ต้องมี `ncnn`) |
| `best.onnx` | สำรองถ้าโหลด NCNN ไม่ได้ (ต้องมี `onnxruntime`) |
| `best.pt` | ต้นฉบับ PyTorch สำรองสุดท้าย |

## ข้อมูลและการเทรน (train.py)

**ข้อมูลที่ใช้เทรนโมเดลที่ส่ง**: ภาพ**กองเมล็ด**ที่ทีมถ่ายเองบนกระดาษขาว (เม็ดชิด/ซ้อน เหมือนลักษณะ test)
วาด polygon ด้วยมือใน Roboflow export แบบ YOLOv8 (ภาพ 432×432) อยู่ใน `dataset_rf/` ของ repo ทีม
- train 27 ภาพ / valid 7 ภาพ, 2 คลาส (`good`, `shell`), valid มี 104 เม็ด

**ขั้นตอน**
1. เริ่มจาก `yolo11n-seg.pt` (COCO) fine-tune สั้น ๆ 15 epoch บนภาพของงานแข่ง 21 ภาพ (circle/grid/square)
   โดยใช้ label อัตโนมัติจาก threshold (`tools/autolabel.py`, `tools/make_yolo.py`) → `yolo11n-seg-beanstart.pt`
2. fine-tune ต่อบนภาพกอง Roboflow: CPU, imgsz 448, batch 8, 40 epoch, mosaic, copy-paste 0.3, หมุน 180°, flip
3. export เป็น ONNX และ NCNN (imgsz 448)

```sh
python train.py --models yolo11n-seg-beanstart.pt --data dataset_rf/beans_rf.yaml --imgsz 448 --epochs 40
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); m.export(format='onnx', imgsz=448, simplify=True); m.export(format='ncnn', imgsz=448)"
```

การเทรนต้องติดตั้งเพิ่ม: `onnx`, `onnxslim`, `pnnx` (สำหรับ export) สคริปต์ใน `tools/` อยู่ใน repo ของทีม

**เลือกรุ่นโมเดล**: เลือก YOLO11n-seg จาก 3 รุ่นในสไลด์ (v8n / 11n / 26n) เพราะบน Pi เร็วพอ ๆ กัน (คะแนน latency เท่ากัน)
และ 11n มี mask mAP50-95 สูงสุด (0.854) ซึ่งสะท้อนความแนบขอบ = คะแนน IoU ทั้งยังเล็กที่สุด

## ผลการทดลอง

คะแนนตามเกณฑ์ Day 2 (สไลด์หน้า 58) คำนวณด้วย `tools/score.py`: จับคู่ mask แบบหนึ่งต่อหนึ่ง (Hungarian, ไม่คิดคลาส)
เม็ดที่ไม่มีคู่และเม็ดที่ทายเกินนับ IoU = 0

**ภาพกอง valid 7 ภาพ (ไม่ได้ใช้เทรน, 104 เม็ด)** · mask mAP50 = 0.95

| ตั้งค่า | IoU (P1 /7.5) | Counting (P2 /7.5) | รวม P /15 |
|---|---|---|---|
| imgsz 448, conf 0.25 (**ที่ส่ง**) | **5.92** | **6.89** | **12.81** |
| imgsz 640, conf 0.25 | 5.56 | 6.52 | 12.09 |
| imgsz 448, conf 0.20 / 0.40 | 5.64 / 5.38 | 6.61 / 6.27 | 12.25 / 11.65 |

- NCNN, ONNX และ `.pt` ให้ผลเท่ากันทุกหลัก (export ไม่ทำให้ความแม่นลด)
- mask IoU เฉลี่ยของเม็ดที่จับคู่ได้ 0.82–0.89 ต่อภาพ

**ความเร็ว** (end-to-end ต่อภาพ: อ่านภาพ → โมเดล → เขียน PNG/JSON/vis, หลัง warm-up)

| เครื่อง | ภาพ valid 432 px | ภาพเต็ม 12 MP (4000×3000) |
|---|---|---|
| โน้ตบุ๊ก Core i5-8265U (CPU) | ~0.2–0.4 s | ~0.50 s |
| Raspberry Pi 5 | (กรอกหลังวัด) | (กรอกหลังวัด) |

## ตรวจผลเอง

```sh
python tools/yolo_to_gt.py dataset_rf/valid val_rf gt_rf     # polygon -> GT PNG
python infer.py --input val_rf --weights weights/ --out outputs/
python tools/score.py --pred outputs --gt-dir gt_rf --latency 0.5
```

## ข้อจำกัด

- โมเดลเห็นเฉพาะคลาส `good` และ `shell` ในภาพกอง เม็ดคลาสอื่นยังถูกหาเจอ (ทดสอบกับภาพเต็มของ black/broken)
  แต่จะถูกทายคลาสเป็น good/shell และมีแนวโน้มนับเกิน (black: จริง 10 → ทาย 16) คะแนน Day 2 ไม่คิดคลาส
- valid มีแค่ 7 ภาพ ตัวเลขข้างบนจึงคลาดเคลื่อนได้
- ภาพ Roboflow ถูกย่อเป็น 432×432 ถ้าตอน export ใช้แบบ stretch รูปทรงเม็ดตอนเทรนจะถูกบีบเล็กน้อยเทียบกับภาพ test จริง
