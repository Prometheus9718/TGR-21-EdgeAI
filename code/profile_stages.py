"""Time each step of infer.predict:  python profile_stages.py [photo_folder]   (default: test_data)"""
import io, os, statistics, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import infer as I

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "test_data")
files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))
models = I.load_models(os.path.join(HERE, "weights")); I.warm_up(models)
sess, cls = models
T = {}


def tick(name, t0):
    T.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
    return time.perf_counter()


for rep in range(3):
    for f in files:
        raw = open(os.path.join(folder, f), "rb").read()
        t = t_all = time.perf_counter()
        half, full_wh = I.load_image(io.BytesIO(raw)); t = tick("1 decode JPEG (half size)", t)
        x0, y0, x1, y1 = I.bean_region(half); crop = half[y0:y1, x0:x1]; t = tick("2 find bean region", t)
        dets = I.run_seg(sess, crop); t = tick("3 segmenter", t)
        ci, cb = I.refine_instances(dets, crop); t = tick("4 refine edges + contacts", t)
        ci, cb = I.apply_f1_filter(ci, cb, list(I.QUALITY_LOG)); t = tick("5 fragment rule", t)
        ids, boxes = I.to_full(ci, cb, (x0, y0), half.shape, full_wh); t = tick("6 masks to full size", t)
        polys = I.polygons_from_ids(ids, boxes); t = tick("7 polygons", t)
        hid = np.zeros(half.shape[:2], np.uint16); hid[y0:y0 + ci.shape[0], x0:x0 + ci.shape[1]] = ci
        I.classify(cls, half, ids, boxes, full_wh, hid); t = tick("8 classifier", t)
        T.setdefault("TOTAL", []).append((time.perf_counter() - t_all) * 1000)
print(f"{len(files)} photos x 3 runs, segmenter {sess.kind}, classifier {cls.kind if cls is not None else '-'} — median ms per photo:")
for k in sorted(T, key=lambda s: (s == "TOTAL", s)):
    print(f"  {k:<28} {statistics.median(T[k]):7.1f}")

if cls is not None and cls.nc is not None:                 # both classifier runtimes on the same photos
    kept = cls.kind
    for kind in ("opencv", "ncnn"):
        cls.kind = kind; ts = []
        for f in files:
            raw = open(os.path.join(folder, f), "rb").read()
            half, full_wh = I.load_image(io.BytesIO(raw)); x0, y0, x1, y1 = I.bean_region(half); crop = half[y0:y1, x0:x1]
            ci, cb = I.refine_instances(I.run_seg(sess, crop), crop); ci, cb = I.apply_f1_filter(ci, cb, list(I.QUALITY_LOG))
            ids, boxes = I.to_full(ci, cb, (x0, y0), half.shape, full_wh)
            hid = np.zeros(half.shape[:2], np.uint16); hid[y0:y0 + ci.shape[0], x0:x0 + ci.shape[1]] = ci
            t = time.perf_counter(); I.classify(cls, half, ids, boxes, full_wh, hid); ts.append((time.perf_counter() - t) * 1000)
        print(f"  classifier on {kind:<6}: {statistics.median(ts):6.1f} ms")
    cls.kind = kept
