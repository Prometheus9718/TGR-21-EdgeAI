"""Tight mixed piles built from the REAL Day 3 beans (real label outlines), full resolution, with ground truth."""
import os, json, cv2, numpy as np, sys
sys.path.insert(0, '/home/claude/day2'); import gen_dataset as g
from PIL import Image
CL = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
LAB = json.load(open('/mnt/user-data/uploads/781b1b2d-labels.json')); out = 'd3piles_tri'; rng = np.random.default_rng(57)
os.makedirs(f'{out}/images', exist_ok=True); os.makedirs(f'{out}/gt', exist_ok=True)
cuts, cls_of, bgs = [], [], []
for e in [x for x in LAB if 'triangle' in x['image_name']]:
    img = g.load_half('/mnt/user-data/uploads/Day3/' + e['image_name']); s = img.shape[1] / 3072.0; masks = []
    for o in e['objects']:
        m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(np.array(o['polygon']) * s).astype(np.int32)], 1); masks.append(m > 0); cls_of.append(o['class'])
    cu, bg = g.cutouts_and_background(img, masks); cuts += cu; bgs.append(bg)
src = open('/home/claude/day2/gen_dataset.py').read(); body = src[src.index('def make_pile'):src.index('def crop_region')]
body = body.replace('rng.uniform(0.92, 1.08) + rng.uniform(-8, 8)', 'rng.uniform(0.97, 1.03) + rng.uniform(-3, 3)')
body = body.replace('patch, mask = cuts[int(rng.integers(len(cuts)))]', 'ci_ = int(rng.integers(len(cuts))); patch, mask = cuts[ci_]; CLS[k] = ci_')
body = body.replace('def make_pile(cuts, bg, rng, n_range=(6, 46)):', 'def make_pile(cuts, bg, rng, n_range=(6, 46)):\n    global CLS\n    CLS = {}')
body = body.replace('return np.clip(canvas, 0, 255).astype(np.uint8), out', 'return np.clip(canvas, 0, 255).astype(np.uint8), out, [CLS[k] for k in keep]')
ns = dict(np=np, cv2=cv2, _rotate=g._rotate); exec(body, ns); labels = {}
for j in range(12):
    nr = [(10, 16), (16, 26), (26, 40)][j % 3]
    pile, ids, src_idx = ns['make_pile'](cuts, bgs[j % len(bgs)], rng, nr)
    full = cv2.resize(pile, (pile.shape[1] * 2, pile.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    fids = cv2.resize(ids.astype(np.uint16), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    name = f'd3pile_{j:02d}'; Image.fromarray(full).save(f'{out}/images/{name}.jpg', quality=92); cv2.imwrite(f'{out}/gt/{name}.png', fids)
    labels[name] = [cls_of[i] for i in src_idx]; print(name, int(fids.max()), 'beans')
json.dump(labels, open(f'{out}/labels.json', 'w'))
