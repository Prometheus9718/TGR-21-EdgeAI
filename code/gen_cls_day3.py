"""Classifier crops from the real labelled Day 3 photos.
real crops: each labelled bean cut from its own photo (other beans painted out), white-balanced.
pile crops: mixed synthetic piles built from the Day 3 bean cutouts (realistic lighting), white-balanced.
Usage: SEED=1 python gen_cls_day3.py <photo1,photo2,...> <n_piles> <out.npz> [real_repeat]"""
import os, sys, json, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gen_dataset as g
CL = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
LAB = {e['image_name']: e for e in json.load(open(os.environ.get('LABELS', 'labels.json')))}
photos = sys.argv[1].split(','); n_piles = int(sys.argv[2]); out = sys.argv[3]; rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = np.random.default_rng(int(os.environ.get('SEED', 1)))
WIN, CROP = 192, 96; REF = np.array([210, 210, 210], np.float32)
def wb(c, paper): return np.clip(c.astype(np.float32) * (REF / np.maximum(paper.astype(np.float32), 1)), 0, 255).astype(np.uint8)
def crop_at(img, ids, k, paper, leave=0.0):
    big = cv2.copyMakeBorder(img, WIN, WIN, WIN, WIN, cv2.BORDER_REPLICATE); bid = cv2.copyMakeBorder(ids, WIN, WIN, WIN, WIN, cv2.BORDER_CONSTANT, value=0)
    ys, xs = np.nonzero(ids == k); cx, cy = xs.mean() + rng.normal(0, 2), ys.mean() + rng.normal(0, 2)
    x0, y0 = int(cx) + WIN - WIN // 2, int(cy) + WIN - WIN // 2
    c = big[y0:y0 + WIN, x0:x0 + WIN].copy(); w = bid[y0:y0 + WIN, x0:x0 + WIN]
    for o in np.unique(w):
        if o not in (0, k) and rng.random() >= leave: c[w == o] = paper
    return wb(cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_AREA), paper)
X, Y = [], []; cuts, cls_of, bgs = [], [], []
for name in photos:
    img = g.load_half(os.path.join(os.environ.get('IMAGES', 'images'), name)); s = img.shape[1] / 3072.0
    ids = np.zeros(img.shape[:2], np.int32); masks = []
    for o in LAB[name]['objects']:
        m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(np.array(o['polygon']) * s).astype(np.int32)], 1)
        ids[m > 0] = o['id']; masks.append(m > 0); cls_of.append(CL.index(o['class']))
    cu, bg = g.cutouts_and_background(img, masks); cuts += cu; bgs.append(bg)
    paper = np.median(img[::4, ::4][ids[::4, ::4] == 0], axis=0).astype(np.uint8)
    for r in range(rep):
        for o in LAB[name]['objects']:
            X.append(crop_at(img, ids, o['id'], paper)); Y.append(CL.index(o['class']))
print('real crops', len(X), 'unique bean instances', len(cuts), flush=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_dataset.py')).read(); body = src[src.index('def make_pile'):src.index('def crop_region')]
body = body.replace('rng.uniform(0.92, 1.08) + rng.uniform(-8, 8)', 'rng.uniform(0.97, 1.03) + rng.uniform(-3, 3)')
body = body.replace('return np.clip(canvas, 0, 255).astype(np.uint8), out', 'canvas = canvas * rng.uniform(0.85, 1.12) * rng.uniform(0.95, 1.05, 3)\n    return np.clip(canvas, 0, 255).astype(np.uint8), out')
body = body.replace('patch, mask = cuts[int(rng.integers(len(cuts)))]', 'ci_ = int(rng.integers(len(cuts))); patch, mask = cuts[ci_]; CLS[k] = ci_')
body = body.replace('def make_pile(cuts, bg, rng, n_range=(6, 46)):', 'def make_pile(cuts, bg, rng, n_range=(6, 46)):\n    global CLS\n    CLS = {}')
body = body.replace('return np.clip(canvas, 0, 255).astype(np.uint8), out', 'return np.clip(canvas, 0, 255).astype(np.uint8), out, [CLS[k] for k in keep]')
ns = dict(np=np, cv2=cv2, _rotate=g._rotate); exec(body, ns); t0 = time.time()
for j in range(n_piles):
    pile, ids, src_idx = ns['make_pile'](cuts, bgs[int(rng.integers(len(bgs)))], rng, (8, 40))
    paper = np.median(pile[::4, ::4][ids[::4, ::4] == 0], axis=0).astype(np.uint8)
    for k, si in enumerate(src_idx, 1):
        if (ids == k).sum() < 300: continue
        X.append(crop_at(pile, ids, k, paper, leave=0.15)); Y.append(cls_of[si])
    if j % 50 == 0: print(j, len(X), f'{time.time()-t0:.0f}s', flush=True)
X = np.stack(X); Y = np.array(Y); np.savez_compressed(out, X=X, Y=Y); print('saved', X.shape, np.bincount(Y, minlength=7))
