"""Per-bean defect classifier on a GPU (falls back to CPU).
Crops: 96x96, touching beans painted out in paper colour, white-balanced to the paper colour.
  python train_cls_gpu.py cv     -> 3-fold check: hold out grid / square / triangle photos in turn (all 124 real beans tested once)
  python train_cls_gpu.py final  -> 3 networks on all 16 photos, averaged -> bean_cls.onnx
Needs: day1_crops.npz (gen_cls_day1.py), Day 3 photos + labels.json (gen_cls_day3.py is called per fold)."""
import os, sys, json, subprocess, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import day1_train as T
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CL = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
EPOCHS = int(os.environ.get('EPOCHS', 25)); PILES = int(os.environ.get('PILES', 350)); REAL_REP = int(os.environ.get('REAL_REP', 12))
LAB = json.load(open(os.environ.get('LABELS', 'labels.json'))); PHOTOS = [e['image_name'] for e in LAB]
def norm(X): return torch.from_numpy(((X.astype(np.float32) / 255 - 0.5) / 0.25).transpose(0, 3, 1, 2).copy())
def augment(b):
    k = random.randint(0, 3); b = torch.rot90(b, k, dims=(2, 3))
    if random.random() < 0.5: b = torch.flip(b, dims=(3,))
    n = b.shape[0]; gain = 1 + (torch.rand(n, 1, 1, 1, device=b.device) - 0.5) * 0.1; bias = (torch.rand(n, 1, 1, 1, device=b.device) - 0.5) * 0.1
    b = b * gain + bias; th = torch.zeros(n, 2, 3, device=b.device); s = 1 + (torch.rand(n, device=b.device) - 0.5) * 0.1
    th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, :, 2] = (torch.rand(n, 2, device=b.device) - 0.5) * 0.1
    return F.grid_sample(b, F.affine_grid(th, b.shape, align_corners=False), padding_mode='border', align_corners=False)
def crops(photos, piles, rep, out, seed):
    env = dict(os.environ, SEED=str(seed))
    subprocess.run([sys.executable, 'gen_cls_day3.py', ','.join(photos), str(piles), out, str(rep)], check=True, env=env, stdout=subprocess.DEVNULL)
    d = np.load(out); return d['X'], d['Y']
def train(X, Y, seed, Xv=None, Yv=None):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = T.BeanCNN().to(dev); Xt = norm(X).to(dev); Yt = torch.from_numpy(Y).long().to(dev); bs = 128; steps = (len(Xt) + bs - 1) // bs
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * steps)
    swa = torch.optim.swa_utils.AveragedModel(m); swa_from = EPOCHS - max(5, EPOCHS // 3)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]; loss = F.cross_entropy(m(augment(Xt[idx])), Yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep >= swa_from: swa.update_parameters(m)
    torch.optim.swa_utils.update_bn([Xt[i:i + 256] for i in range(0, len(Xt), 256)], swa)
    return swa.module.eval()
def predict(models, X):
    with torch.no_grad():
        Xt = norm(X).to(dev); return torch.stack([torch.cat([torch.softmax(m(Xt[i:i + 256]), 1) for i in range(0, len(Xt), 256)]) for m in models]).mean(0).argmax(1).cpu().numpy()
def day1():
    d = np.load('day1_crops.npz'); rng = np.random.default_rng(0); i = rng.choice(len(d['Y']), min(6000, len(d['Y'])), replace=False); return d['X'][i], d['Y'][i]
if __name__ == '__main__':
    mode = sys.argv[1]; X1, Y1 = day1(); t0 = time.time()
    if mode == 'cv':
        ok = n = 0; per = np.zeros((7, 2), int)
        for fold in ['grid', 'square', 'triangle']:
            tr = [p for p in PHOTOS if fold not in p.lower()]; te = [p for p in PHOTOS if fold in p.lower()]
            Xa, Ya = crops(tr, PILES, REAL_REP, f'cls_{fold}_train.npz', 1); Xv, Yv = crops(te, 0, 1, f'cls_{fold}_test.npz', 2)
            mdl = train(np.concatenate([Xa, X1]), np.concatenate([Ya, Y1]), 4)
            p = predict([mdl], Xv); ok += (p == Yv).sum(); n += len(Yv)
            for t, q in zip(Yv, p): per[t, 0] += t == q; per[t, 1] += 1
            print(f'fold {fold:8s}: {(p == Yv).sum()}/{len(Yv)} = {(p == Yv).mean():.0%}   ({time.time() - t0:.0f}s)', flush=True)
        print(f'CROSS-CHECK on all real beans: {ok}/{n} = {ok / n:.1%}   by type: ' + ', '.join(f'{CL[i]} {per[i, 0]}/{per[i, 1]}' for i in range(7)), flush=True)
    else:
        Xa, Ya = crops(PHOTOS, int(PILES * 1.3), REAL_REP, 'cls_all_train.npz', 8); X = np.concatenate([Xa, X1]); Y = np.concatenate([Ya, Y1])
        models = [train(X, Y, s) for s in (11, 12, 13)]
        class Ens(nn.Module):
            def __init__(s, ms): super().__init__(); s.ms = nn.ModuleList(ms)
            def forward(s, x): return torch.stack([torch.softmax(m(x), 1) for m in s.ms]).mean(0)
        e = Ens([m.cpu() for m in models]).eval()
        torch.onnx.export(e, torch.zeros(1, 3, 96, 96), 'bean_cls.onnx', input_names=['input'], output_names=['probs'], dynamic_axes={'input': {0: 'batch'}, 'probs': {0: 'batch'}}, opset_version=17, dynamo=False)
        print(f'exported bean_cls.onnx (3 networks, {EPOCHS} epochs each)  {time.time() - t0:.0f}s', flush=True)
