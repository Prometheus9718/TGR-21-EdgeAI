"""Day 2 dataset generator: YOLO-seg labels from real photos + synthetic piles.

Real training photos have separated beans, so OpenCV gives perfect per-bean masks.
Test photos are piles (beans touch and overlap). We cut every bean out with its mask
and paste beans into piles on the inpainted paper; an id map records which bean is
visible at each pixel, so labels are the VISIBLE part of each bean (like real labels).

Each saved image is a crop around the beans (same as infer.py does at test time),
at "half resolution" (photo width 1536 px).

    python gen_dataset.py --data day_1/ --out beans_ds --piles 18 [--val-layout square]
"""
import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

import day1_infer as d1

CLASSES = d1.CLASSES
HALF_W = 1536


def load_half(path):
    """Decode at ~half size (fast JPEG draft mode), apply EXIF, resize to HALF_W wide."""
    with Image.open(path) as im:
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        a = np.asarray(im)
    s = HALF_W / a.shape[1]
    return cv2.resize(a, (HALF_W, int(round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)


def tight_mask(patch, loose, chroma_t=7.0, dark_t=0.62):
    """Shrink a loose bean mask to the bean's real edge, removing its grey shadow.

    A shadow has the paper's colour (only darker); a bean has its own colour (brown,
    yellow, green) or is very dark. Pixel = bean if its colour differs from the local
    paper colour (Lab chroma distance) or it is much darker than the paper.
    """
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = (cv2.dilate(loose, np.ones((31, 31), np.uint8)) > 0) & (loose == 0)
    pa = np.median(lab[ring], axis=0) if ring.sum() > 50 else np.array([215, 128, 128], np.float32)
    chroma = np.sqrt((lab[..., 1] - pa[1]) ** 2 + (lab[..., 2] - pa[2]) ** 2)
    m = ((chroma > chroma_t) | (lab[..., 0] < dark_t * pa[0])) & (loose > 0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab_ == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def bean_instances(img):
    """Separated beans on paper -> list of TIGHT boolean masks (bean edge, no shadow)."""
    lo = cv2.resize(img, (d1.WORK_W, int(img.shape[0] * d1.WORK_W / img.shape[1])), interpolation=cv2.INTER_AREA)
    m = d1.bean_mask(lo)
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab = cv2.connectedComponents(m)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        y0, y1 = max(0, ys.min() - 25), min(img.shape[0], ys.max() + 26)
        x0, x1 = max(0, xs.min() - 25), min(img.shape[1], xs.max() + 26)
        t = tight_mask(img[y0:y1, x0:x1], (lab[y0:y1, x0:x1] == i).astype(np.uint8))
        if t.sum() < 300:
            continue
        full = np.zeros(img.shape[:2], bool)
        full[y0:y1, x0:x1] = t > 0
        out.append(full)
    return out


def cutouts_and_background(img, masks):
    cuts = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        p = 6
        y0, x0 = max(0, y0 - p), max(0, x0 - p)
        y1, x1 = min(img.shape[0], y1 + p), min(img.shape[1], x1 + p)
        cuts.append((img[y0:y1, x0:x1].copy(), (m[y0:y1, x0:x1] * 255).astype(np.uint8)))
    small = cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    paper = cv2.medianBlur(cv2.medianBlur(small, 21), 21)              # beans + shadows removed
    bg = cv2.resize(paper, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    noise = np.random.default_rng(len(masks)).normal(0, 2.0, img.shape[:2])[..., None]
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cuts, bg


def _rotate(patch, mask, angle, flip, scale):
    if flip:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    h, w = mask.shape
    side = int(np.hypot(h, w) * scale) + 4
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    R[:, 2] += (side / 2 - w / 2, side / 2 - h / 2)
    P = cv2.warpAffine(np.ascontiguousarray(patch), R, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    M = cv2.warpAffine(np.ascontiguousarray(mask), R, (side, side), flags=cv2.INTER_NEAREST)
    return P, M


def make_pile(cuts, bg, rng, n_range=(6, 46)):
    """Paste n beans into a pile. Returns image and int32 id map (0 = paper, k = bean k)."""
    canvas = bg.astype(np.float32).copy()
    H, W = canvas.shape[:2]
    ids = np.zeros((H, W), np.int32)
    full_area = {}
    n = int(rng.integers(*n_range))
    size = np.median([max(m.shape) for _, m in cuts])
    sigma = size * rng.uniform(0.14, 0.30) * np.sqrt(n)      # tight: beans touch and overlap
    cx = W / 2 + rng.normal(0, W * 0.04); cy = H / 2 + rng.normal(0, H * 0.04)
    for k in range(1, n + 1):
        patch, mask = cuts[int(rng.integers(len(cuts)))]
        P, M = _rotate(patch, mask, float(rng.uniform(0, 360)), bool(rng.integers(2)), float(rng.uniform(0.93, 1.07)))
        # small colour jitter per bean (lighting differences)
        P = np.clip(P.astype(np.float32) * rng.uniform(0.92, 1.08) + rng.uniform(-8, 8), 0, 255)
        h, w = M.shape
        x = int(np.clip(cx + rng.normal(0, sigma) - w / 2, 0, W - w))
        y = int(np.clip(cy + rng.normal(0, sigma) - h / 2, 0, H - h))
        roi = canvas[y:y + h, x:x + w]
        sh = cv2.GaussianBlur(cv2.dilate(M, np.ones((7, 7), np.uint8)), (15, 15), 0)
        sh = np.roll(sh, (int(rng.integers(6, 14)), int(rng.integers(-3, 4))), axis=(0, 1)).astype(np.float32)[..., None] / 255
        roi *= (1 - rng.uniform(0.18, 0.32) * sh)
        a = cv2.GaussianBlur(M, (3, 3), 0).astype(np.float32)[..., None] / 255
        roi[:] = roi * (1 - a) + P * a
        ids[y:y + h, x:x + w][M > 127] = k
        full_area[k] = int((M > 127).sum())
    keep = [k for k in full_area if (ids == k).sum() >= max(250, 0.25 * full_area[k])]
    out = np.zeros_like(ids)
    for j, k in enumerate(keep, 1):
        out[ids == k] = j
    return np.clip(canvas, 0, 255).astype(np.uint8), out


def crop_region(img, ids, rng=None, margin=80):
    ys, xs = np.nonzero(ids)
    m = margin if rng is None else int(rng.integers(40, 160))
    y0, y1 = max(0, ys.min() - m), min(img.shape[0], ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(img.shape[1], xs.max() + m)
    return img[y0:y1, x0:x1], ids[y0:y1, x0:x1]


def polygons(ids):
    H, W = ids.shape
    out = []
    for k in range(1, ids.max() + 1):
        m = (ids == k).astype(np.uint8)
        if m.sum() < 50:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(c.astype(np.float32) / [W, H])
    return out


def save(img, ids, name, split, out):
    cv2.imwrite(f"{out}/images/{split}/{name}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(f"{out}/gt/{split}/{name}.npy", ids.astype(np.uint16))
    with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
        for poly in polygons(ids):
            f.write("0 " + " ".join(f"{v:.5f}" for v in poly.flatten()) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="beans_ds")
    ap.add_argument("--piles", type=int, default=18, help="synthetic piles per photo")
    ap.add_argument("--mixed", type=int, default=60, help="extra piles mixing all classes")
    ap.add_argument("--val-layout", default="", help="e.g. 'square': its photos' piles go to val only")
    ap.add_argument("--big-frac", type=float, default=0.0, help="fraction of piles with 46-120 beans")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for s in ("train", "val"):
        for d in ("images", "labels", "gt"):
            os.makedirs(f"{a.out}/{d}/{s}", exist_ok=True)
    all_cuts = {"train": [], "val": []}
    bgs = []
    for cls in CLASSES:
        for p in sorted(glob.glob(os.path.join(a.data, cls, "*"))):
            name = os.path.splitext(os.path.basename(p))[0]
            split = "val" if a.val_layout and name.startswith(a.val_layout) else "train"
            img = load_half(p)
            masks = bean_instances(img)
            ids = np.zeros(img.shape[:2], np.int32)
            for k, m in enumerate(masks, 1):
                ids[m] = k
            ci, cid = crop_region(img, ids)
            save(ci, cid, f"real_{name}", split, a.out)
            cuts, bg = cutouts_and_background(img, masks)
            all_cuts[split] += cuts
            bgs.append(bg)
            for j in range(a.piles):
                nr = (46, 121) if rng.random() < a.big_frac else (6, 46)
                pile, pid = make_pile(cuts, bg, rng, nr)
                pi, pidc = crop_region(pile, pid, rng)
                save(pi, pidc, f"pile_{name}_{j:02d}", split, a.out)
            print(f"{split:5s} {name:22s} {len(masks)} beans -> {a.piles} piles", flush=True)
    for j in range(a.mixed):               # mixed-class piles (Day 3 style), train split only
        bg = bgs[int(rng.integers(len(bgs)))]
        nr = (46, 121) if rng.random() < a.big_frac else (6, 46)
        pile, pid = make_pile(all_cuts["train"], bg, rng, nr)
        pi, pidc = crop_region(pile, pid, rng)
        save(pi, pidc, f"mixed_{j:03d}", "train", a.out)
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: bean\n")
    for s in ("train", "val"):
        print(s, len(os.listdir(f"{a.out}/images/{s}")), "images")


if __name__ == "__main__":
    main()
