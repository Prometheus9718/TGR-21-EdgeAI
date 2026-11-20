"""Day 3 segmentation dataset (one class "bean", YOLO polygons):
  - the real Day 3 photos with their hand-drawn outlines (labels.json),
  - tight mixed piles built from the real Day 3 beans (their real outlines),
  - Day 1 photos / piles (colour-rule outlines) and mixed Day 1 + Day 3 piles for variety.
Validation = the held-out layout (default: triangle) — its real photos and piles of its beans only.

python gen_dataset_d3.py --day1 day_1 --day3 day3 --labels labels.json --out beans_d3
"""
import argparse, glob, json, os
import cv2, numpy as np
import gen_dataset as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day1", default="day_1"); ap.add_argument("--day3", default="day3"); ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out", default="beans_d3"); ap.add_argument("--val-layout", default="triangle")
    ap.add_argument("--d3-piles", type=int, default=900); ap.add_argument("--mixed", type=int, default=200)
    ap.add_argument("--d1-piles", type=int, default=10); ap.add_argument("--val-piles", type=int, default=60)
    ap.add_argument("--real-repeat", type=int, default=4); ap.add_argument("--big-frac", type=float, default=0.25)
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    for s in ("train", "val"):
        for d in ("images", "labels", "gt"):
            os.makedirs(f"{a.out}/{d}/{s}", exist_ok=True)
    def pile_range():
        return (46, 91) if rng.random() < a.big_frac else (6, 46)
    # ---- Day 3: real photos with hand-drawn outlines
    lab = json.load(open(a.labels))
    d3cuts = {"train": [], "val": []}; d3bgs = {"train": [], "val": []}
    for e in lab:
        stem = os.path.splitext(e["image_name"])[0]
        split = "val" if a.val_layout and a.val_layout in stem.lower() else "train"
        img = g.load_half(os.path.join(a.day3, e["image_name"])); s = img.shape[1] / 3072.0
        ids = np.zeros(img.shape[:2], np.int32); masks = []
        for k, o in enumerate(e["objects"], 1):
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(np.array(o["polygon"], np.float32) * s).astype(np.int32)], 1)
            ids[(m > 0) & (ids == 0)] = k; masks.append(m > 0)
        for r in range(a.real_repeat if split == "train" else 1):
            ci, cid = g.crop_region(img, ids, rng if r else None)
            g.save(ci, cid, f"d3real_{stem}_{r}", split, a.out)
        cuts, bg = g.cutouts_and_background(img, masks); d3cuts[split] += cuts; d3bgs[split].append(bg)
        print(f"day3 {split:5s} {stem:18s} {len(masks)} beans", flush=True)
    # ---- Day 1: photos (colour-rule outlines) and a few piles each
    d1cuts, d1bgs = [], []
    for p in sorted(glob.glob(os.path.join(a.day1, "*", "*.jpg"))):
        name = os.path.splitext(os.path.basename(p))[0]
        img = g.load_half(p); masks = g.bean_instances(img)
        if not masks:
            continue
        ids = np.zeros(img.shape[:2], np.int32)
        for k, m in enumerate(masks, 1):
            ids[m] = k
        ci, cid = g.crop_region(img, ids); g.save(ci, cid, f"d1real_{name}", "train", a.out)
        cuts, bg = g.cutouts_and_background(img, masks); d1cuts += cuts; d1bgs.append(bg)
        for j in range(a.d1_piles):
            pile, pid = g.make_pile(cuts, bg, rng, pile_range()); pi, pidc = g.crop_region(pile, pid, rng)
            g.save(pi, pidc, f"d1pile_{name}_{j:02d}", "train", a.out)
        print(f"day1 train {name:22s} {len(masks)} beans", flush=True)
    # ---- tight mixed piles from the real Day 3 beans (+ Day 3 paper)
    for j in range(a.d3_piles):
        bg = d3bgs["train"][int(rng.integers(len(d3bgs["train"])))]
        pile, pid = g.make_pile(d3cuts["train"], bg, rng, pile_range()); pi, pidc = g.crop_region(pile, pid, rng)
        g.save(pi, pidc, f"d3pile_{j:04d}", "train", a.out)
        if j % 100 == 0:
            print("day3 piles", j, flush=True)
    # ---- mixed Day 1 + Day 3 piles
    allc = d1cuts + d3cuts["train"]; allb = d1bgs + d3bgs["train"]
    for j in range(a.mixed):
        pile, pid = g.make_pile(allc, allb[int(rng.integers(len(allb)))], rng, pile_range()); pi, pidc = g.crop_region(pile, pid, rng)
        g.save(pi, pidc, f"mix_{j:04d}", "train", a.out)
    # ---- validation piles from the held-out layout's beans
    for j in range(a.val_piles):
        bg = d3bgs["val"][int(rng.integers(len(d3bgs["val"])))]
        pile, pid = g.make_pile(d3cuts["val"], bg, rng, pile_range()); pi, pidc = g.crop_region(pile, pid, rng)
        g.save(pi, pidc, f"d3valpile_{j:03d}", "val", a.out)
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: bean\n")
    for s in ("train", "val"):
        print(s, len(os.listdir(f"{a.out}/images/{s}")), "images", flush=True)


if __name__ == "__main__":
    main()
