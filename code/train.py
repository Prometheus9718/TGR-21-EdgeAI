"""TGR-21 Day 3 segmenter training: YOLO26n-seg, one class "bean". Run on a GPU (we used Colab, A100:
colab/day3_segmenter.ipynb). The weights in weights/ were made in two stages.

  Stage 1 (Day 2 model, from COCO weights):
    python gen_dataset.py --data day_1/ --out beans_ds --piles 25 --mixed 80 --val-layout square
    python train.py --data beans_ds/data.yaml --model yolo26n-seg.pt --imgsz 640 --epochs 150 \
                    --close-mosaic 15 --patience 50 --name day2

  Stage 2 (Day 3 fine-tune, the model in weights/):
    python gen_dataset_d3.py --day1 day_1 --day3 day3 --labels labels.json --out beans_d3 \
           --d3-piles 900 --mixed 200 --d1-piles 10 --val-piles 60 --real-repeat 4 --big-frac 0.25
    python train.py --data beans_d3/data.yaml --model runs/day2/weights/best.pt --imgsz 896 --epochs 60 \
                    --close-mosaic 10 --patience 25 --name day3 --export-imgsz 640

Stage 2 data: the 16 labelled Day 3 photos (hand-drawn outlines, labels.json), tight mixed piles built from their
beans (6-90 beans, touching and overlapping; a bean less than 25% visible is left out of the labels), Day 1 photos
and piles; validation = the triangle layout's photos and piles (held out).
The model is trained at 896 and exported at 640 for the Pi: on held-out piles 640 scores within 0.13 / 30 of 896
and runs ~1.8x faster. Exports: weights/best.pt, weights/seg.onnx (ONNX Runtime), weights/seg_ncnn_model/ (NCNN).

The bean classifier (weights/bean_cls.onnx) is trained by train_cls_gpu.py — see README.md.
"""
import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Train / fine-tune the bean segmenter and export it for the Pi")
    ap.add_argument("--data", default="beans_d3/data.yaml")
    ap.add_argument("--model", default="runs/day2/weights/best.pt", help="starting weights (yolo26n-seg.pt for stage 1)")
    ap.add_argument("--imgsz", type=int, default=896, help="training size")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default="day3")
    ap.add_argument("--export-imgsz", type=int, default=640, help="input size of the exported model (infer.IMGSZ)")
    ap.add_argument("--out", default="weights")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=8, cache="ram",
                degrees=180, flipud=0.5, fliplr=0.5, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
                mosaic=1.0, close_mosaic=a.close_mosaic, patience=a.patience, project="runs", name=a.name, exist_ok=True)
    best = os.path.join("runs", a.name, "weights", "best.pt")
    m = YOLO(best)
    met = m.val(data=a.data, imgsz=a.imgsz)
    print("mask mAP50", met.seg.map50, " mask mAP50-95", met.seg.map)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(best, os.path.join(a.out, "best.pt"))
    shutil.copy(m.export(format="onnx", imgsz=a.export_imgsz, simplify=True), os.path.join(a.out, "seg.onnx"))
    ncnn_dir = m.export(format="ncnn", imgsz=a.export_imgsz)
    dst = os.path.join(a.out, "seg_ncnn_model")
    os.makedirs(dst, exist_ok=True)
    for f in ("model.ncnn.param", "model.ncnn.bin"):
        shutil.copy(os.path.join(ncnn_dir, f), dst)
    print("saved", a.out)


if __name__ == "__main__":
    main()
