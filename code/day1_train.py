"""Train a small CNN on single-bean crops, then export it to ONNX for the Raspberry Pi.

Usage:
    python train.py --data path/to/day_1 --out weights/best.onnx            # train on everything + export
    python train.py --data path/to/day_1 --cv                               # leave-one-layout-out check

Each training photo holds ~8 beans of one class. beans.py cuts them out, so the
CNN learns from ~170 bean crops instead of 21 photos. At inference the beans in a
photo vote (mean of their softmax probabilities) to give one class per photo.
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from day1_infer import crop_features
from day1_infer import CLASSES, CROP, bean_centers, bean_mask, crop_beans, crops_at, load_rgb, work_image  # same preprocessing as inference

SEED = 0
JITTER = 0.0   # brightness/contrast augmentation strength
PILES_TRAIN = 6  # synthetic piles per training photo (training crops)
PILES_VAL = 4    # synthetic piles per photo kept for validation only
ARCH = "cnn"     # "cnn" = small CNN from scratch, "mnv3" = fine-tuned MobileNetV3-Small (set by --arch)
PRETRAINED = True
MEAN = np.array([0.5, 0.5, 0.5], np.float32)   # must match infer.py
STD = np.array([0.25, 0.25, 0.25], np.float32)


# ---------------------------------------------------------------- synthetic piles
# Test photos are natural piles (beans touch and overlap) but training photos are
# arranged. We cut every bean out of a training photo, inpaint the paper behind it,
# and paste rotated beans close together to make realistic piles (slide 20, tip 6).
def cutouts_and_background(rgb):
    img = work_image(rgb)
    mask = bean_mask(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    cuts = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        p = 6
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(img.shape[1], x + w + p), min(img.shape[0], y + h + p)
        m = (lab[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        cuts.append((img[y0:y1, x0:x1].copy(), m))
    hole = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hole_s = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    bg_s = cv2.inpaint(small, hole_s, 7, cv2.INPAINT_TELEA)
    bg = cv2.resize(bg_s, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    bg = np.where(hole[..., None] > 0, bg, img)
    return cuts, bg


def _rotate(patch, mask, angle, flip):
    if flip:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    h, w = mask.shape
    side = int(np.hypot(h, w)) + 2
    px, py = (side - w) // 2, (side - h) // 2
    P = cv2.copyMakeBorder(np.ascontiguousarray(patch), py, side - h - py, px, side - w - px, cv2.BORDER_REPLICATE)
    M = cv2.copyMakeBorder(np.ascontiguousarray(mask), py, side - h - py, px, side - w - px, cv2.BORDER_CONSTANT, value=0)
    R = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    P = cv2.warpAffine(P, R, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    M = cv2.warpAffine(M, R, (side, side), flags=cv2.INTER_NEAREST)
    return P, M


def make_pile(cuts, bg, rng, n_beans=None, spread=0.55):
    """Paste n beans (sampled with replacement) into a tight, overlapping pile on bg."""
    canvas = bg.astype(np.float32).copy()
    H, W = canvas.shape[:2]
    n_beans = n_beans or int(rng.integers(7, 13))
    size = np.median([max(m.shape) for _, m in cuts])
    cx = W / 2 + rng.normal(0, W * 0.05)
    cy = H / 2 + rng.normal(0, H * 0.05)
    for _ in range(n_beans):
        patch, mask = cuts[int(rng.integers(len(cuts)))]
        P, M = _rotate(patch, mask, float(rng.uniform(0, 360)), bool(rng.integers(2)))
        x = int(cx + rng.normal(0, size * spread * 1.2) - P.shape[1] / 2)
        y = int(cy + rng.normal(0, size * spread * 1.2) - P.shape[0] / 2)
        x = int(np.clip(x, 0, W - P.shape[1])); y = int(np.clip(y, 0, H - P.shape[0]))
        h, w = M.shape
        roi = canvas[y:y + h, x:x + w]
        # soft contact shadow under the bean
        sh = cv2.GaussianBlur(cv2.dilate(M, np.ones((7, 7), np.uint8)), (9, 9), 0).astype(np.float32) / 255
        roi *= (1 - 0.25 * sh[..., None])
        a = cv2.GaussianBlur(M, (3, 3), 0).astype(np.float32)[..., None] / 255
        roi[:] = roi * (1 - a) + P.astype(np.float32) * a
    return np.clip(canvas, 0, 255).astype(np.uint8), n_beans


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class BeanCNN(nn.Module):
    """4 conv blocks (conv-BN-ReLU-maxpool) + global average pooling + dropout + linear."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))            # global average pooling
        return self.head(x)


def to_tensor(crops):
    x = (np.stack(crops).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def augment(batch):
    """Random 90-degree rotations, flips, small brightness/contrast change. Training only."""
    k = random.randint(0, 3)
    batch = torch.rot90(batch, k, dims=(2, 3))
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=(3,))
    b = batch.shape[0]
    gain = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * JITTER
    bias = (torch.rand(b, 1, 1, 1) - 0.5) * JITTER
    batch = batch * gain + bias
    # small random scale + shift via affine grid
    theta = torch.zeros(b, 2, 3)
    s = 1 + (torch.rand(b) - 0.5) * 0.1      # small: bean size is a useful cue
    theta[:, 0, 0] = s; theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(b, 2) - 0.5) * 0.1
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    return F.grid_sample(batch, grid, padding_mode="border", align_corners=False)


def load_dataset(root, train_piles=PILES_TRAIN, val_piles=PILES_VAL):
    """Return a list of dicts: path, class index, layout, real crops, synthetic pile crops.

    Training uses the real crops plus crops from synthetic piles made from the same
    photo's beans. Validation uses separate synthetic piles (never used for training).
    """
    rng = np.random.default_rng(SEED)
    items = []
    for ci, cls in enumerate(CLASSES):
        for p in sorted(glob.glob(os.path.join(root, cls, "*"))):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rgb = load_rgb(p)
            crops = crop_beans(rgb)
            cuts, bg = cutouts_and_background(rgb)
            pile_crops = []
            for _ in range(train_piles):
                img, _n = make_pile(cuts, bg, rng)
                pile_crops += crops_at(img, bean_centers(bean_mask(img)))
            val = []
            for _ in range(val_piles):
                img, _n = make_pile(cuts, bg, rng)
                val.append(crops_at(img, bean_centers(bean_mask(img))))
            layout = os.path.basename(p).split("_")[0]
            items.append(dict(path=p, ci=ci, layout=layout, crops=crops, piles=pile_crops, val_piles=val))
            print(f"{cls:10s} {os.path.basename(p):28s} {len(crops)} beans, {len(pile_crops)} pile crops")
    return items


class MobileNetV3Bean(nn.Module):
    """MobileNetV3-Small (ImageNet weights) fine-tuned for 7 bean classes (slides 14, 18, 21).

    Takes the SAME input as BeanCNN (96x96, normalised with MEAN/STD), so infer.py does
    not change: inside, it undoes that normalisation, upsamples to 160 px and applies
    ImageNet normalisation before the backbone.
    """

    def __init__(self, pretrained=True, size=160):
        super().__init__()
        import torchvision as tv
        m = tv.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
        self.net, self.size = m, size
        self.register_buffer("m_in", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s_in", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("m_im", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s_im", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.s_in + self.m_in                                  # back to 0..1 RGB
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net((x - self.m_im) / self.s_im)


def _batches(X, y, bs, device):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        idx = perm[i:i + bs]
        yield augment(X[idx]).to(device), y[idx].to(device)


def train_model(items, epochs=60, verbose=True, seed=SEED):
    set_seed(seed)
    crops = [c for it in items for c in it["crops"] + it["piles"]]
    labels = [it["ci"] for it in items for _ in it["crops"] + it["piles"]]
    X, y = to_tensor(crops), torch.tensor(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = (len(X) + 31) // 32

    if ARCH == "mnv3":
        # stage 1: freeze the ImageNet backbone, train only the new head (slide 14)
        model = MobileNetV3Bean(pretrained=PRETRAINED).to(device)
        freeze = min(5, epochs // 4)
        model.net.features.requires_grad_(False)
        opt = torch.optim.AdamW(model.net.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = None
    else:
        model = BeanCNN().to(device)
        freeze = 0
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps)

    for ep in range(epochs):
        if ARCH == "mnv3" and ep == freeze:
            # stage 2: unfreeze everything, small lr for the backbone, cosine decay
            model.net.features.requires_grad_(True)
            opt = torch.optim.AdamW([{"params": model.net.features.parameters(), "lr": 1e-4},
                                     {"params": model.net.classifier.parameters(), "lr": 1e-3}],
                                    weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(epochs - freeze) * steps)
        model.train()
        tot, correct = 0.0, 0
        for xb, yb in _batches(X, y, 32, device):
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item() * len(yb); correct += (out.argmax(1) == yb).sum().item()
        if verbose and (ep % 10 == 9 or ep == 0):
            print(f"epoch {ep + 1:3d}  loss {tot / len(X):.3f}  crop acc {correct / len(X):.2f}")
    return model.cpu().eval()


class Ensemble(nn.Module):
    """Several CNNs trained with different seeds; outputs their averaged class probabilities."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([F.softmax(m(x), dim=1) for m in self.models]).mean(0)


def train_ensemble(items, epochs, n_models, verbose=True):
    return Ensemble([train_model(items, epochs, verbose, seed=SEED + i) for i in range(n_models)])


@torch.no_grad()
def predict_photo(model, crops):
    model.eval()
    probs = model(to_tensor(crops)).mean(0)   # each bean votes with its probabilities
    return int(probs.argmax()), probs


def cross_validate(items, epochs, n_models):
    """Train on two layouts, test on the third (circle / grid / square).

    Reports accuracy on the held-out arranged photos AND on synthetic piles built
    from the held-out photos' beans (closer to the real test set).
    """
    layouts = sorted({it["layout"] for it in items})
    real = [0, 0]; pile = [0, 0]
    saved = []   # (true class, mean probs, kind) for tuning the class boost
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    for held in layouts:
        tr = [it for it in items if it["layout"] != held]
        te = [it for it in items if it["layout"] == held]
        model = train_ensemble(tr, epochs, n_models, verbose=False).eval()
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xf = np.array([crop_features(c) for it in tr for c in it["crops"] + it["piles"]])
        yf = np.array([it["ci"] for it in tr for _ in it["crops"] + it["piles"]])
        sc = StandardScaler().fit(Xf)
        lr = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Xf), yf)
        feat_p = lambda cs: lr.predict_proba(sc.transform(np.array([crop_features(c) for c in cs])))
        for it in te:
            pred, pr = predict_photo(model, it["crops"])
            with torch.no_grad():
                saved.append((it["ci"], model(to_tensor(it["crops"])).numpy(), "arranged", feat_p(it["crops"])))
                for c in it["val_piles"]:
                    if c:
                        saved.append((it["ci"], model(to_tensor(c)).numpy(), "pile", feat_p(c)))
            real[0] += pred == it["ci"]; real[1] += 1
            pp = [predict_photo(model, c)[0] if c else -1 for c in it["val_piles"]]
            for q in pp:
                pile[0] += q == it["ci"]; pile[1] += 1
                if q >= 0:
                    confusion[it["ci"], q] += 1
            print(f"[{held:6s}] {os.path.basename(it['path']):26s} true={CLASSES[it['ci']]:9s} "
                  f"arranged={CLASSES[pred]:9s} piles={[CLASSES[q] if q >= 0 else '-' for q in pp]}")
    print(f"\nArranged photos: {real[0]}/{real[1]} = {real[0] / real[1]:.2f}")
    print(f"Synthetic piles: {pile[0]}/{pile[1]} = {pile[0] / pile[1]:.2f}")
    np.save("cv_probs.npy", np.array(saved, dtype=object), allow_pickle=True)
    print("pile confusion (rows=true, cols=pred):", CLASSES)
    for ci, row in enumerate(confusion):
        print(f"  {CLASSES[ci]:9s}", row)


def train_color_model(items, path):
    """Logistic regression on colour/size features of every training crop -> JSON for infer.py."""
    import json
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array([crop_features(c) for it in items for c in it["crops"] + it["piles"]])
    y = np.array([it["ci"] for it in items for _ in it["crops"] + it["piles"]])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X), y)
    json.dump({"classes": [CLASSES[i] for i in lr.classes_], "mean": sc.mean_.tolist(),
               "scale": sc.scale_.tolist(), "coef": lr.coef_.tolist(),
               "intercept": lr.intercept_.tolist()}, open(path, "w"), indent=1)
    print("saved", path, "train acc", round(lr.score(sc.transform(X), y), 3))


def export_onnx(model, path):
    model.eval()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dummy = torch.zeros(1, 3, CROP, CROP)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["probs"],
                      dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    print("exported", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one subfolder per class")
    ap.add_argument("--out", default="weights/best.onnx")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", type=int, default=3, help="number of CNNs in the ensemble")
    ap.add_argument("--cv", action="store_true", help="only run leave-one-layout-out validation")
    ap.add_argument("--color-only", action="store_true", help="only (re)train the colour/size model")
    ap.add_argument("--arch", choices=["cnn", "mnv3"], default="cnn",
                    help="cnn = small CNN from scratch; mnv3 = MobileNetV3-Small fine-tuned from ImageNet")
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only) mnv3 without ImageNet weights")
    a = ap.parse_args()
    ARCH = a.arch
    PRETRAINED = not a.no_pretrained

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    data = load_dataset(a.data)
    if a.cv:
        cross_validate(data, a.epochs, a.models)
    else:
        color_path = os.path.join(os.path.dirname(a.out) or ".", "color_lr.json")
        train_color_model(data, color_path)
        if not a.color_only:
            m = train_ensemble(data, a.epochs, a.models)
            export_onnx(m, a.out)
