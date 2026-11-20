"""Day 1 inference: classify each photo of green coffee beans into one of 7 classes.

    python infer.py --input test/ --weights weights/ --out outputs/

Pipeline per photo (fixed preprocessing, no test-time augmentation):
  1. read the image and apply EXIF orientation
  2. resize to 1024 px wide, segment beans from the white paper with OpenCV
     (background-flattened threshold + saturation mask), then find one centre per
     bean with distance-transform peaks, which also works when beans touch in a pile
  3. cut a fixed 128x128 window around every centre (keeps true bean size), resize to 96x96
  4. a CNN ensemble (ONNX, CPU) gives class probabilities for every bean
  4b. a second model (logistic regression on colour + size of each bean, color_lr.json)
     gives its own probabilities; they are blended 65% CNN / 35% colour-size
  5. majority vote: every bean votes for its class (immature probability weighted, see
     CLASS_WEIGHT); the class with the most votes is the photo's class. Ties are broken by
     the summed probability of the tied classes. Each photo is decided on its own.
If no bean is found, the whole image is classified instead.
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
EXTS = (".jpg", ".jpeg", ".png")
CROP = 96             # CNN input size
WORK_W = 1024         # working width for bean detection (beans are ~60-90 px wide here)
WIN = 128             # fixed window around each bean centre, in working pixels
MIN_AREA = 800        # smallest blob that can be a bean (working pixels)
NMS_R = 26            # minimum distance between two bean centres
MIN_DIST = 14         # a centre must be at least this far inside the bean mask
MEAN, STD = 0.5, 0.25  # input normalisation (same as train.py)
# Decision weights applied to the averaged class probabilities. The CNN rarely says
# "immature" but is rarely wrong when it does, so its score is boosted x3 (final check
# on fresh held-out piles: immature 12/24 -> 17/24, piles 155 -> 157/168, arranged 19 -> 20/21). Tuned on
# leave-one-layout-out validation: synthetic piles 76/84 -> 79/84, arranged 19/21 -> 20/21,
# and no other class got worse (a boost above ~2.1 starts to break other classes).
CLASS_WEIGHT = {"immature": 3.0}
# Second model: logistic regression on colour + size features of each bean
# (weights/color_lr.json, saved by train.py next to best.onnx). Per-bean probabilities are
# blended: (1 - COLOR_MIX) * CNN + COLOR_MIX * colour/size. Validation: piles 79/84 -> 82/84,
# arranged 18/21 -> 21/21 (majority vote, immature x2).
COLOR_MIX = 0.35


# ---------------------------------------------------------------- preprocessing
def load_rgb(path):
    """Read an image, apply EXIF orientation, return an RGB uint8 array."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def bean_mask(img):
    """Binary mask (uint8 0/255) of bean pixels on white paper."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 101).astype(np.float32) + 1          # uneven lighting estimate
    flat = np.clip(gray.astype(np.float32) / bg * 200, 0, 255)      # paper ~200, beans darker
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    mask = (((flat < 165) | (sat > 60)) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # keep bean-sized or bigger blobs (a pile is one big blob); drop noise, lines, border junk
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA or area > 0.4 * h * w:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 6:
            continue
        keep[i] = True
    return (keep[lab] * 255).astype(np.uint8)


def bean_centers(mask):
    """Approximate bean centres: local maxima of the distance transform.

    Works for separated beans (one peak each) and for touching / piled beans,
    where the peaks fall inside the individual beans of the merged blob.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    k = 2 * NMS_R + 1
    peaks = (dist >= cv2.dilate(dist, np.ones((k, k), np.uint8))) & (dist >= MIN_DIST)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-dist[ys, xs])
    centers = []
    for i in order:                                   # greedy non-maximum suppression
        p = (int(xs[i]), int(ys[i]))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > NMS_R ** 2 for q in centers):
            centers.append(p)
    return centers


def work_image(rgb):
    scale = WORK_W / rgb.shape[1]
    return cv2.resize(rgb, (WORK_W, max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def crops_at(img, centers):
    """Fixed-size WIN x WIN windows around each centre (keeps real bean size), resized to CROP."""
    pad = WIN
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    out = []
    for x, y in centers:
        x0, y0 = x + pad - WIN // 2, y + pad - WIN // 2
        out.append(cv2.resize(big[y0:y0 + WIN, x0:x0 + WIN], (CROP, CROP), interpolation=cv2.INTER_AREA))
    return out


def crop_beans(rgb):
    """List of CROPxCROP RGB crops, one per detected bean centre."""
    img = work_image(rgb)
    return crops_at(img, bean_centers(bean_mask(img)))


def crop_features(crop):
    """Colour + size measurements of the bean(s) in one CROPxCROP crop (15 numbers).

    Bean pixels = darker than the paper or clearly coloured. Size = distance from the
    crop centre to the nearest bean edge (half the bean width, in working pixels),
    which still works when neighbouring beans touch.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper = np.percentile(gray, 95) + 1
    m = (gray < 0.82 * paper) | (hsv[..., 1] > 60)
    if m.sum() < 20:
        m = np.ones_like(m)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    c = CROP // 2
    half_w = dist[c - 3:c + 4, c - 3:c + 4].max() * WIN / CROP
    L, A, B = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    S, V = hsv[..., 1][m], hsv[..., 2][m]
    rel = gray[m] / paper                                   # brightness relative to paper
    return np.array([half_w, L.mean(), L.std(), np.percentile(L, 10), np.percentile(L, 90),
                     A.mean(), B.mean(), B.std(), np.percentile(B, 90), S.mean(), V.mean(),
                     rel.mean(), np.percentile(rel, 10), np.percentile(rel, 90),
                     (B - A).mean()], np.float32)


def to_input(crops):
    """List of HxWx3 uint8 crops -> float32 NCHW batch, normalised."""
    x = (np.stack(crops).astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------- model
class OnnxNet:
    """Runs the ONNX CNN with OpenCV's DNN module (no onnxruntime needed on the Pi).

    Verified identical to onnxruntime: max output difference 4e-7, same decision for every bean.
    """

    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)

    def run(self, x):
        self.net.setInput(x)
        return self.net.forward()


def find_weights(weights_arg):
    """--weights is the weights FOLDER (organiser update). Find the CNN .onnx inside it.

    Prefers best.onnx; otherwise the only .onnx file. A direct path to an .onnx file
    also works. Returns the path of the .onnx file.
    """
    if os.path.isfile(weights_arg):
        return weights_arg
    if not os.path.isdir(weights_arg):
        fail(f"weights folder not found: {weights_arg}")
    onnx = sorted(f for f in os.listdir(weights_arg) if f.lower().endswith(".onnx"))
    if not onnx:
        fail(f"no .onnx model in weights folder {weights_arg}")
    name = "best.onnx" if "best.onnx" in onnx else onnx[0]
    return os.path.join(weights_arg, name)


def load_model(weights_path):
    """Returns (ONNX CNN session, colour/size model or None).

    The colour/size model is color_lr.json, in the same folder as the ONNX file.
    """
    import json
    sess = OnnxNet(weights_path)
    color_path = os.path.join(os.path.dirname(os.path.abspath(weights_path)), "color_lr.json")
    color_model = None
    if COLOR_MIX > 0:
        if not os.path.isfile(color_path):
            raise FileNotFoundError(f"missing {color_path} (colour/size model, saved by train.py)")
        c = json.load(open(color_path))
        color_model = {k: (np.array(v, np.float32) if k != "classes" else v) for k, v in c.items()}
    return sess, color_model


def color_probs(color_model, crops):
    """Per-bean class probabilities from the colour/size logistic regression (numpy only)."""
    X = np.array([crop_features(c) for c in crops], np.float32)
    z = (X - color_model["mean"]) / color_model["scale"]
    logits = z @ color_model["coef"].T + color_model["intercept"]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    out = np.zeros((len(crops), len(CLASSES)), np.float32)       # reorder to CLASSES
    for j, c in enumerate(color_model["classes"]):
        out[:, CLASSES.index(c)] = p[:, j]
    return out


def bean_probs(model, image_path):
    """Class probabilities for every bean in one photo: array (n_beans, 7)."""
    sess, color_model = model
    rgb = load_rgb(image_path)
    crops = crop_beans(rgb)
    if not crops:                                    # fallback: classify the whole photo
        crops = [cv2.resize(rgb, (CROP, CROP), interpolation=cv2.INTER_AREA)]
    P = sess.run(to_input(crops))
    if color_model is not None:
        P = (1 - COLOR_MIX) * P + COLOR_MIX * color_probs(color_model, crops)
    return P


def majority_vote(P):
    """Each bean votes for its class; the class with the most votes is the photo's class.

    Ties are broken by the summed UNweighted probability of the tied classes, so the
    immature boost can never win a tie on its own (fresh-pile check: piles 157 -> 158/168).
    """
    weights = np.array([CLASS_WEIGHT.get(c, 1.0) for c in CLASSES], np.float32)
    W = P * weights
    votes = np.bincount(W.argmax(axis=1), minlength=len(CLASSES))
    tied = np.flatnonzero(votes == votes.max())
    return CLASSES[int(tied[P.sum(axis=0)[tied].argmax()])]


def predict_image(model, image_path):
    """Return the predicted class name for one photo (majority vote of its beans)."""
    return majority_vote(bean_probs(model, image_path))


# ---------------------------------------------------------------- main
def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def list_images(inp):
    if os.path.isfile(inp):
        if not inp.lower().endswith(EXTS):
            fail(f"not a supported image file: {inp}")
        return [inp]
    if not os.path.isdir(inp):
        fail(f"input not found: {inp}")
    files = sorted(os.path.join(inp, f) for f in os.listdir(inp)
                   if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(inp, f)))
    if not files:
        fail(f"no .jpg/.jpeg/.png images in {inp}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Day 1 coffee bean photo classifier")
    ap.add_argument("--input", required=True, help="folder of test images, or one image file")
    ap.add_argument("--weights", required=True,
                    help="weights folder (weights/) holding best.onnx + color_lr.json; a path to the .onnx also works")
    ap.add_argument("--out", required=True, help="output folder (created if missing)")
    a = ap.parse_args()

    weights_file = find_weights(a.weights)
    images = list_images(a.input)

    in_dir = os.path.abspath(a.input if os.path.isdir(a.input) else os.path.dirname(a.input))
    out_dir = os.path.abspath(a.out)
    if out_dir == in_dir:
        fail("--out must be different from the input folder")
    names = [os.path.basename(p) for p in images]
    if len(set(names)) != len(names):
        fail("duplicate input file names")

    try:
        model = load_model(weights_file)
    except Exception as e:
        fail(f"could not load weights from {a.weights}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    classes = []
    for p in images:
        try:
            classes.append(predict_image(model, p))
        except Exception as e:
            fail(f"could not process {p}: {e}")

    rows = []
    for p, cls in zip(images, classes):
        if cls not in CLASSES:
            fail(f"invalid class {cls!r} for {p}")
        shutil.copy2(p, os.path.join(out_dir, cls, os.path.basename(p)))
        rows.append((os.path.basename(p), cls))
        print(f"{os.path.basename(p)},{cls}")

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class"])
        w.writerows(rows)
    if len(rows) != len(images):
        fail("prediction count does not match image count")
    print(f"done: {len(rows)} images -> {os.path.join(out_dir, 'predictions.csv')}")


if __name__ == "__main__":
    main()
