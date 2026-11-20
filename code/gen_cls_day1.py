"""Usage: SEED=21 python gen_cls_data.py circle,grid 300 cls_train.npz day_1   (and: square 40 cls_val_sq.npz for the check set)
Realistic lighting: small per-bean colour change, one colour cast per photo; crops white-balanced to the paper colour.
Training crops for a per-bean classifier: mixed-class synthetic piles, neighbours painted out."""
import sys, time, cv2, numpy as np
import gen_dataset as g
layouts = sys.argv[1].split(','); n_piles = int(sys.argv[2]); out = sys.argv[3]
DATA = sys.argv[4] if len(sys.argv) > 4 else 'day_1'   # folder with one subfolder per class
import os; rng = np.random.default_rng(int(os.environ.get('SEED', 5)))
WIN, CROP = 192, 96
REF = np.array([210, 210, 210], np.float32)   # paper is mapped to this colour (white balance)            # classifier window at half resolution (128 px at 1024 wide), resized to 96
cuts, cls_of, bgs = [], [], []
for ci, c in enumerate(g.CLASSES):
    for lay in layouts:
        img = g.load_half(f'{DATA}/{c}/{lay}_{c}.jpg')
        cu, bg = g.cutouts_and_background(img, g.bean_instances(img)); cuts += cu; cls_of += [ci] * len(cu); bgs.append(bg)
print('unique beans', len(cuts), 'per class', np.bincount(cls_of), flush=True)
src = open('gen_dataset.py').read(); body = src[src.index('def make_pile'):src.index('def crop_region')]
body = body.replace('rng.uniform(0.92, 1.08) + rng.uniform(-8, 8)', 'rng.uniform(0.97, 1.03) + rng.uniform(-3, 3)')
body = body.replace('return np.clip(canvas, 0, 255).astype(np.uint8), out', 'canvas = canvas * rng.uniform(0.85, 1.12) * rng.uniform(0.95, 1.05, 3)\n    return np.clip(canvas, 0, 255).astype(np.uint8), out')
body = body.replace('patch, mask = cuts[int(rng.integers(len(cuts)))]', 'ci_ = int(rng.integers(len(cuts))); patch, mask = cuts[ci_]; CL[k] = ci_')
body = body.replace('def make_pile(cuts, bg, rng, n_range=(6, 46)):', 'def make_pile(cuts, bg, rng, n_range=(6, 46)):\n    global CL\n    CL = {}')
body = body.replace('return np.clip(canvas, 0, 255).astype(np.uint8), out', 'return np.clip(canvas, 0, 255).astype(np.uint8), out, [CL[k] for k in keep]')
ns = dict(np=np, cv2=cv2, _rotate=g._rotate); exec(body, ns)
X, Y = [], []; t0 = time.time()
for j in range(n_piles):
    pile, ids, src_idx = ns['make_pile'](cuts, bgs[int(rng.integers(len(bgs)))], rng, (12, 70))
    paper = np.median(pile[::4, ::4][ids[::4, ::4] == 0], axis=0).astype(np.uint8)
    big = cv2.copyMakeBorder(pile, WIN, WIN, WIN, WIN, cv2.BORDER_REPLICATE)
    bid = cv2.copyMakeBorder(ids, WIN, WIN, WIN, WIN, cv2.BORDER_CONSTANT, value=0)
    for k, si in enumerate(src_idx, 1):
        ys, xs = np.nonzero(ids == k)
        if len(ys) < 300: continue
        cx, cy = xs.mean() + rng.normal(0, 2), ys.mean() + rng.normal(0, 2)
        x0, y0 = int(cx) + WIN - WIN // 2, int(cy) + WIN - WIN // 2
        c = big[y0:y0 + WIN, x0:x0 + WIN].copy(); w = bid[y0:y0 + WIN, x0:x0 + WIN]
        others = [o for o in np.unique(w) if o not in (0, k)]
        # paint neighbours with paper; leave ~15% visible to mimic beans the segmenter missed
        for o in others:
            if rng.random() > 0.15: c[w == o] = paper
        r = cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32) * (REF / np.maximum(paper.astype(np.float32), 1))
        X.append(np.clip(r, 0, 255).astype(np.uint8)); Y.append(cls_of[si])
    if j % 50 == 0: print(j, len(X), f'{time.time()-t0:.0f}s', flush=True)
X = np.stack(X); Y = np.array(Y); np.savez_compressed(out, X=X, Y=Y); print('saved', X.shape, np.bincount(Y))
