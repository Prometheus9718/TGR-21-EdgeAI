"""Check Day 3 outputs against the rules: python check_outputs.py <input_folder> <output_folder>"""
import os, sys, json, numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import image_size                      # also reads files Pillow rejects (e.g. a PNG with a broken checksum)
CL = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
inp, out = sys.argv[1], sys.argv[2]; errs = []
imgs = sorted(f for f in os.listdir(inp) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
pred = {e['image_name']: e for e in json.load(open(os.path.join(out, 'predictions.json'), encoding='utf-8'))}
for f in imgs:
    stem = os.path.splitext(f)[0]
    wh = image_size(os.path.join(inp, f))
    if wh is None:
        e = pred.get(f)
        if e is None or e['objects']: errs.append(f'{f}: unreadable photo but answer is not empty')
        print(f'   note: {f} is not a readable image (answered with 0 beans)'); continue
    W, H = wh
    mp = os.path.join(out, 'seg_mask', stem + '.png')
    if not os.path.exists(mp): errs.append(f'{f}: no seg_mask'); continue
    m = np.array(Image.open(mp))
    if m.ndim != 2 or m.shape != (H, W): errs.append(f'{f}: mask shape {m.shape} != {(H, W)}')
    e = pred.get(f)
    if e is None: errs.append(f'{f}: missing in predictions.json'); continue
    n = len(e['objects']); ids = sorted(set(np.unique(m).tolist()) - {0})
    if ids != list(range(1, n + 1)): errs.append(f'{f}: mask ids {ids[:5]}... vs {n} objects')
    for o in e['objects']:
        if o['class'] not in CL: errs.append(f'{f}: bad class {o["class"]}')
        p = np.array(o['polygon'])
        if len(p) < 3 or p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max() >= W or p[:, 1].max() >= H: errs.append(f'{f}: bad polygon for id {o["id"]}')
    q = json.load(open(os.path.join(out, 'mqtt', stem + '.json'), encoding='utf-8'))
    cnt = {c: sum(o['class'] == c for o in e['objects']) for c in CL}
    if q.get('image') != f: errs.append(f'{f}: mqtt image field {q.get("image")}')
    if list(q['counts'].keys()) != CL or q['counts'] != cnt: errs.append(f'{f}: mqtt counts {q["counts"]} != predictions {cnt}')
    dr = round((n - cnt['good']) / n, 4) if n else 0.0
    if abs(q['defect_ratio'] - dr) > 1e-6 or not isinstance(q['latency_ms'], int): errs.append(f'{f}: defect_ratio/latency {q["defect_ratio"]} {q["latency_ms"]}')
    if not os.path.exists(os.path.join(out, 'vis', stem + '.jpg')): errs.append(f'{f}: no vis')
print(f'{len(imgs)} photos checked: ' + ('ALL OK' if not errs else f'{len(errs)} problems'))
for x in errs[:20]: print('  ', x)
sys.exit(1 if errs else 0)
