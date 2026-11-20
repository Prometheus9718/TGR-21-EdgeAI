"""Score the final build (NCNN, 640) on every set in the shareable dataset pack, judge-style."""
import sys, os, json, cv2, numpy as np
os.environ['TGR_RUNTIME'] = 'ncnn'
sys.path.insert(0, '/home/claude/day2'); from score import iou_matrix
sys.path.insert(0, '/home/claude/day3/final_repo/TGR-21_Day3'); import infer as I
from scipy.optimize import linear_sum_assignment
models = I.load_models('/home/claude/day3/final_repo/TGR-21_Day3/weights'); I.warm_up(models)
D = '/home/claude/share/TGR-21_EdgeAI_dataset'
def score(name, items):
    T = dict(photos=0, gt=0, pred=0, found=0, hi=0, ious=0.0, denom=0, ok=0, n=0)
    for path, gt_fn, labs in items:
        p = I.predict(path, models); gt = gt_fn(p['ids'].shape)
        M = iou_matrix(gt, p['ids']); ng, npd = M.shape; r, c = linear_sum_assignment(-M); v = M[r, c]
        T['photos'] += 1; T['gt'] += ng; T['pred'] += npd; T['found'] += int((v > 0.5).sum()); T['hi'] += int((v > 0.9).sum())
        T['ious'] += float(v.sum()); T['denom'] += max(ng, npd)
        for gi, pi, vv in zip(r, c, v):
            if vv > 0.5: T['n'] += 1; T['ok'] += p['classes'][pi] == labs[gi]
    print(f"{name:28s} photos {T['photos']:2d}  beans {T['gt']:4d}  ours {T['pred']:4d}  found {T['found']/T['gt']:.1%}  "
          f"IoU>0.90 {T['hi']/T['gt']:.1%}  mean IoU {T['ious']/T['denom']:.3f}  class {T['ok']/T['n']:.1%} ({T['ok']}/{T['n']})", flush=True)
LAB = json.load(open('/mnt/user-data/uploads/781b1b2d-labels.json')); real = {'grid': [], 'square': [], 'triangle': []}
for e in LAB:
    objs = e['objects']
    def gt_fn(shape, objs=objs):
        g = np.zeros(shape, np.uint16)
        for k, o in enumerate(objs, 1): cv2.fillPoly(g, [np.round(np.array(o['polygon'])).astype(np.int32)], k)
        return g
    kind = [k for k in real if k in e['image_name']][0]
    real[kind].append(('/mnt/user-data/uploads/Day3/' + e['image_name'], gt_fn, [o['class'] for o in objs]))
score('real 16 photos', real['grid'] + real['square'] + real['triangle']); score('real triangle 5 (seg held out)', real['triangle'])
def from_dir(img_dir, gt_dir, lab_path, names=None):
    lab = json.load(open(lab_path)); out = []
    for nm in sorted(names or lab):
        out.append((f'{img_dir}/{nm}.jpg', lambda shape, p=f'{gt_dir}/{nm}.png': cv2.imread(p, -1), lab[nm]))
    return out
for s in ['random_piles', 'triangle_piles']:
    score(s, from_dir(f'{D}/simulated_piles/{s}/images', f'{D}/simulated_piles/{s}/gt', f'{D}/simulated_piles/{s}/labels.json'))
lab = json.load(open(f'{D}/stress_tests/labels.json'))
score('stress 100-155 beans', from_dir(f'{D}/stress_tests', f'{D}/stress_tests', f'{D}/stress_tests/labels.json', [k for k in lab if not k.startswith('far')]))
score('stress camera farther back', from_dir(f'{D}/stress_tests', f'{D}/stress_tests', f'{D}/stress_tests/labels.json', [k for k in lab if k.startswith('far')]))
