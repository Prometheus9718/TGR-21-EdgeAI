"""TGR-21 · TESA Top Gun Rally 2026 · Edge AI Day 3 (Integration): every coffee bean in a mixed pile,
its mask, its outline and its class, on a Raspberry Pi 5 (CPU).

    python infer.py --input <test_folder/> --weights <weights_folder/> --out <output_folder/>
    (client.py runs the same pipeline on photos it receives over HTTP)

outputs/
  seg_mask/<photo>.png   full size, uint16: 0 = paper, 1..N = bean id
  predictions.json       per photo: image_name + objects [{id, class, polygon [[x, y], ...]}] in full-photo pixels
  mqtt/<photo>.json      the per-photo MQTT message: image, latency_ms, counts (7 classes), defect_ratio
  vis/<photo>.jpg        quarter-size overlay coloured by class, with each bean's id and class

Pipeline per photo:
  1. JPEG decoded at half size (draft mode) + EXIF orientation; StreamDecoder does it while the photo downloads
  2. the bean region is found with a cheap low-res threshold and cropped
  3. YOLO26n-seg (one class "bean", run at 640; NCNN on the Pi, ONNX Runtime elsewhere) -> boxes + mask logits
  4. edges refined: the model decides which bean a pixel belongs to, a colour rule decides where the bean/paper
     edge is (shadows excluded), a watershed on the photo re-draws the boundary where two beans touch;
     every bean becomes one solid piece, so mask and polygon agree
  5. every bean is kept; only fragments under 0.3 x the median bean area are dropped
  6. class per bean: a small CNN on a 96x96 crop with touching beans painted out and the crop white-balanced
     to the paper colour; NCNN or OpenCV DNN, whichever is faster on the machine
"""
import argparse
import io
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = (".jpg", ".jpeg", ".png")
CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
DEFECTS = [c for c in CLASSES if c != "good"]   # defect_ratio = defective beans / all beans
# overlay colours per class (RGB)
CLASS_RGB = {"good": (40, 200, 40), "shell": (255, 160, 0), "black": (90, 90, 255), "husk": (200, 0, 200),
             "broken": (255, 40, 40), "parchment": (0, 200, 200), "immature": (255, 255, 0)}
HALF_W = 1536          # working resolution (same as training crops)
IMGSZ = 640            # YOLO input size
CONF = 0.35            # minimum detection confidence
REFINE = True          # snap bean/paper edges to the colour boundary
CHROMA_T, DARK_T = 7.0, 0.62   # colour rule (same as the training labels)
FG_LOGIT = 1.5         # a bean-coloured pixel joins the most likely bean if its logit > -FG_LOGIT
SURE_LOGIT = 9.0       # only extremely sure model pixels override the colour rule (removes box-shaped spill)
BOX_MARGIN = 0         # crop-space pixels a mask may extend beyond its box
MIN_PIX = 150          # smallest bean (crop pixels at half resolution)
WS_BAND = 3            # re-draw bean-to-bean boundaries within this many pixels of a contact (0 = off)
MIN_VISIBLE = 0.3      # drop a bean if less than 30% of its predicted mask stays visible
QUALITY_LOG = []       # per-bean quality features of the last photo (read by the fragment rule)
# Keep every bean: a missed bean scores nothing. Only fragments under 0.3 x the photo's median bean area are
# dropped (the smallest real labelled bean is 0.55 x). The feature names are those QUALITY_LOG records.
F1_FILTER = {"size": 0.3}
CLS_CROP, CLS_WORK_W, CLS_WIN = 96, 1024, 128      # classifier crop: 96 px, cut from a 128-px window at 1024-px width
CLS_WB = True             # white-balance classifier crops to the paper colour (as in training)
CLS_REF = np.array([210, 210, 210], np.float32)
CLS_MAX_BEANS = 400       # every bean gets its own label; beyond this many, the rest take the photo majority
cv2.setNumThreads(4)


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- image + region
def _raw_bytes(src):
    if hasattr(src, "getvalue"):
        return src.getvalue()
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    with open(src, "rb") as f:
        return f.read()


def image_size(src):
    """(W, H) after EXIF orientation, or None. Works on files Pillow rejects (e.g. a PNG with a broken checksum)."""
    data = _raw_bytes(src)
    try:
        with Image.open(io.BytesIO(data)) as im:
            return ImageOps.exif_transpose(im).size
    except Exception:                                             # noqa: BLE001
        pass
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if a is not None:
        return a.shape[1], a.shape[0]
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:     # read width/height straight from the PNG header
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def load_image(path):
    """Returns (half-size RGB array, full (W, H) after EXIF orientation)."""
    try:
        return _load_image_pil(path)
    except Exception:                                             # noqa: BLE001 — Pillow cannot read it: try OpenCV
        a = cv2.imdecode(np.frombuffer(_raw_bytes(path), np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            raise
        W, H = a.shape[1], a.shape[0]
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        s = HALF_W / W
        a = cv2.resize(a, (HALF_W, max(1, int(round(H * s)))), interpolation=cv2.INTER_AREA)
        return a, (W, H)


def _load_image_pil(path):
    if hasattr(path, "seek"):
        path.seek(0)
    with Image.open(path) as im:
        W, H = im.size
        orient = (im.getexif() or {}).get(274, 1)
        if orient in (5, 6, 7, 8):
            W, H = H, W
        im.draft("RGB", (im.width // 2, im.height // 2))
        return _finish_pil(im, W, H, orient)


def _finish_pil(im, W, H, orient=1):
    if orient not in (0, 1) or im.mode != "RGB":       # rotate / convert only when needed: each copy is ~9 MB
        im = ImageOps.exif_transpose(im).convert("RGB")
    a = np.asarray(im)
    s = HALF_W / a.shape[1]
    if abs(s - 1) > 1e-3:
        a = cv2.resize(a, (HALF_W, int(round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return a, (W, H)


class StreamDecoder:
    """Decodes a JPEG while it is still downloading: feed() each piece as it arrives, result() at the end.
    The server's clock runs during the download, so the decode (~60 ms on the Pi) is then mostly hidden.
    It makes the same Pillow/libjpeg calls as load_image (Pillow itself feeds the decoder in 64 KB pieces),
    so the pixels are identical; anything unusual (not a JPEG, broken, other Pillow version) gives None
    and the caller falls back to load_image on the full bytes."""

    def __init__(self):
        self.buf = bytearray()
        self.data = b""
        self.im = self.decoder = self.wh = None
        self.done = self.bad = False

    def feed(self, chunk):
        self.buf += chunk
        if self.bad or self.done:
            return
        try:
            if self.decoder is None:
                self._start()
            else:
                self.data += chunk
            if self.decoder is not None and self.data:
                n, err = self.decoder.decode(self.data)
                if n < 0:                                   # image complete (err 0) or broken (err < 0)
                    self.done, self.bad, self.data = True, err < 0, b""
                    self._cleanup()
                else:
                    self.data = self.data[n:]
        except Exception:                                   # noqa: BLE001
            self.bad = True
            self._cleanup()

    def _start(self):
        if len(self.buf) < 2048:
            return
        try:
            fp = io.BytesIO(bytes(self.buf))
            im = Image.open(fp)
            W, H = im.size
        except Exception:                                   # noqa: BLE001 — header not complete yet
            if len(self.buf) > 4_000_000:
                self.bad = True
            return
        if im.format != "JPEG" or len(im.tile) != 1:
            self.bad = True
            return
        orient = (im.getexif() or {}).get(274, 1)
        if orient in (5, 6, 7, 8):
            W, H = H, W
        self.orient = orient
        im.draft("RGB", (im.width // 2, im.height // 2))
        im.load_prepare()
        d, e, o, a = im.tile[0]
        im.tile = []
        dec = Image._getdecoder(im.mode, d, a, im.decoderconfig)
        dec.setimage(im.im, e)
        self.decoder, self.im, self.wh = dec, im, (W, H)
        self.data = bytes(self.buf)[o:]

    def _cleanup(self):
        if self.decoder is not None:
            try:
                self.decoder.cleanup()
            except Exception:                               # noqa: BLE001
                pass

    def result(self):
        """(half-size RGB array, (W, H)) exactly as load_image gives, or None."""
        if self.bad or not self.done or self.im is None:
            return None
        try:
            return _finish_pil(self.im, *self.wh, self.orient)
        except Exception:                                   # noqa: BLE001
            return None


def stream_self_test():
    """True if piece-by-piece decoding gives exactly what load_image gives on this machine (run once at start)."""
    try:
        rng = np.random.default_rng(1)
        img = cv2.resize(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), (2048, 1536), interpolation=cv2.INTER_CUBIC)
        b = io.BytesIO()
        Image.fromarray(img).save(b, "JPEG", quality=92)
        data = b.getvalue()
        d = StreamDecoder()
        for i in range(0, len(data), 9000):
            d.feed(data[i:i + 9000])
        got, ref = d.result(), load_image(io.BytesIO(data))
        return got is not None and got[1] == ref[1] and np.array_equal(got[0], ref[0])
    except Exception:                                       # noqa: BLE001
        return False


def bean_region(img):
    """Bounding box (x0, y0, x1, y1) of all beans in the half-size image, with margin."""
    f = 384 / img.shape[1]
    small = cv2.resize(img, (384, int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 41).astype(np.float32) + 1
    flat = gray / bg
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    m = (((flat < 0.82) | (sat > 60)) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    keep = [i for i in range(1, n) if st[i, 4] >= 40 and not (st[i, 0] <= 1 or st[i, 1] <= 1 or
            st[i, 0] + st[i, 2] >= w - 1 or st[i, 1] + st[i, 3] >= h - 1)]
    if not keep:
        return 0, 0, img.shape[1], img.shape[0]
    x0 = min(st[i, 0] for i in keep); y0 = min(st[i, 1] for i in keep)
    x1 = max(st[i, 0] + st[i, 2] for i in keep); y1 = max(st[i, 1] + st[i, 3] for i in keep)
    pad = 100 * f
    x0, y0 = max(0, int((x0 - pad) / f)), max(0, int((y0 - pad) / f))
    x1, y1 = min(img.shape[1], int((x1 + pad) / f)), min(img.shape[0], int((y1 + pad) / f))
    return x0, y0, x1, y1


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


# ---------------------------------------------------------------- models
class SegNet:
    """YOLO-seg runner with the same outputs for both runtimes:
    NCNN on ARM (Raspberry Pi: ~1.85x faster than ONNX Runtime per the Day 2 slides),
    ONNX Runtime elsewhere or if ncnn is not installed. Override: TGR_RUNTIME=onnx|ncnn."""

    def __init__(self, weights_dir):
        import platform
        ncnn_dir = os.path.join(weights_dir, "seg_ncnn_model")
        onnx_path = os.path.join(weights_dir, "seg.onnx")
        want = os.environ.get("TGR_RUNTIME", "ncnn" if platform.machine().lower() in ("aarch64", "arm64", "armv8l") else "onnx")
        self.kind = None
        if want == "ncnn" and os.path.isfile(os.path.join(ncnn_dir, "model.ncnn.param")):
            try:
                import ncnn
                net = ncnn.Net()
                net.opt.use_vulkan_compute = False
                net.opt.num_threads = 4
                net.load_param(os.path.join(ncnn_dir, "model.ncnn.param"))
                net.load_model(os.path.join(ncnn_dir, "model.ncnn.bin"))
                self.ncnn, self.net, self.kind = ncnn, net, "ncnn"
            except Exception as e:                                   # fall back to ONNX
                print(f"note: NCNN unavailable ({e}); using ONNX Runtime", file=sys.stderr)
        if self.kind is None:
            if not os.path.isfile(onnx_path):
                fail(f"missing {onnx_path} (YOLO-seg model)")
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 4
            self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.kind = "onnx"

    def run(self, x):
        if self.kind == "ncnn":
            ex = self.net.create_extractor()
            ex.input("in0", self.ncnn.Mat(x, batch_index=0).clone())
            _, o0 = ex.extract("out0"); _, o1 = ex.extract("out1")
            a, b = o0.numpy(batch_index=0), o1.numpy(batch_index=0)
            return [a.reshape((1,) + a.shape[-2:]), b.reshape((1,) + b.shape[-3:])]
        return self.sess.run(None, {self.sess.get_inputs()[0].name: x})


class ClsRunner:
    """The bean classifier through OpenCV DNN (all beans in one batch) or NCNN (one bean at a time).
    warm_up() times both on this machine and keeps the faster; TGR_CLS_RUNTIME=opencv|ncnn forces one.
    Same network (weights/bean_cls.onnx, converted with pnnx): the labels agree on 540 of 541 real beans."""

    def __init__(self, weights_dir):
        self.cv = cv2.dnn.readNetFromONNX(os.path.join(weights_dir, "bean_cls.onnx"))
        self.nc, self.kind = None, "opencv"
        p = os.path.join(weights_dir, "bean_cls_ncnn", "bean_cls.ncnn.param")
        if os.path.isfile(p) and os.environ.get("TGR_CLS_RUNTIME", "auto") != "opencv":
            try:
                import ncnn
                net = ncnn.Net()
                net.opt.use_vulkan_compute = False
                net.opt.num_threads = 4
                net.load_param(p)
                net.load_model(p[:-len("param")] + "bin")
                self.ncnn, self.nc = ncnn, net
            except Exception:                                   # noqa: BLE001 — keep OpenCV
                self.nc = None

    def forward(self, x):
        if self.kind == "ncnn":
            try:
                out = []
                for xi in x:
                    ex = self.nc.create_extractor()
                    ex.input("in0", self.ncnn.Mat(np.ascontiguousarray(xi)))
                    _, o = ex.extract("out0")
                    out.append(np.array(o, np.float32).reshape(-1))
                return np.stack(out)
            except Exception:                                   # noqa: BLE001 — never fail a photo over this
                self.kind = "opencv"
        self.cv.setInput(x)
        return self.cv.forward()

    def pick_fastest(self):
        x = np.random.default_rng(0).normal(0, 1, (9, 3, CLS_CROP, CLS_CROP)).astype(np.float32)
        self.kind = "opencv"
        self.forward(x)
        if self.nc is None:
            return
        want = os.environ.get("TGR_CLS_RUNTIME", "auto")
        times = {}
        for kind in ("opencv", "ncnn"):
            self.kind = kind
            self.forward(x)
            t = time.perf_counter()
            for _ in range(3):
                p = self.forward(x)
            times[kind] = time.perf_counter() - t
            if kind == "ncnn" and (p.shape != (9, len(CLASSES)) or not np.isfinite(p).all()):
                times[kind] = float("inf")
        self.kind = want if want in times and times[want] < float("inf") else min(times, key=times.get)


def load_models(weights_dir):
    if os.path.isfile(weights_dir):
        weights_dir = os.path.dirname(weights_dir)
    if not os.path.isdir(weights_dir):
        fail(f"weights folder not found: {weights_dir}")
    seg = SegNet(weights_dir)
    seg.weights_dir = weights_dir
    cls_path = os.path.join(weights_dir, "bean_cls.onnx")
    if os.path.isfile(cls_path):
        cls = ClsRunner(weights_dir)
    else:
        cls = None
    return seg, cls


def run_seg(sess, crop):
    """YOLO-seg on one crop -> list of (conf, box[x0,y0,x1,y1] in crop px, mask prob patch for box)."""
    lb, r, left, top = letterbox(crop, IMGSZ)
    x = np.ascontiguousarray((lb.astype(np.float32) / 255).transpose(2, 0, 1)[None])
    outs = sess.run(x)
    det, proto = outs[0][0], outs[1][0]                 # proto: (32, 160, 160)
    if det.shape[0] < det.shape[1]:                    # raw head (4 + nc + 32, 8400) -> NMS
        det = det.T
        nc = det.shape[1] - 4 - proto.shape[0]
        boxes_c, scores = det[:, :4], det[:, 4:4 + nc].max(1)
        coeffs = det[:, 4 + nc:]
        keep = scores > CONF
        boxes_c, scores, coeffs = boxes_c[keep], scores[keep], coeffs[keep]
        xyxy = np.c_[boxes_c[:, :2] - boxes_c[:, 2:] / 2, boxes_c[:, :2] + boxes_c[:, 2:] / 2]
        if len(scores):
            idx = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]].tolist(), scores.tolist(), CONF, 0.6)
            idx = np.array(idx, dtype=np.int64).reshape(-1)
        else:
            idx = np.zeros(0, np.int64)
        xyxy, scores, coeffs = xyxy[idx], scores[idx], coeffs[idx]
    else:                                               # end-to-end (YOLO26): (300, 6 + 32)
        keep = det[:, 4] > CONF
        det = det[keep]
        xyxy, scores, coeffs = det[:, :4], det[:, 4], det[:, 6:]
    nP, ph, pw = proto.shape
    s = ph / IMGSZ
    out = []
    order = np.argsort(-scores)
    for i in order:
        bx0, by0, bx1, by1 = xyxy[i]
        px0, py0 = max(0, int(bx0 * s)), max(0, int(by0 * s))
        px1, py1 = min(pw, int(np.ceil(bx1 * s)) + 1), min(ph, int(np.ceil(by1 * s)) + 1)
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue
        logit = np.tensordot(coeffs[i], proto[:, py0:py1, px0:px1], axes=1)
        # back to crop coordinates
        cx0, cy0 = (px0 / s - left) / r, (py0 / s - top) / r
        cx1, cy1 = (px1 / s - left) / r, (py1 / s - top) / r
        box = ((bx0 - left) / r, (by0 - top) / r, (bx1 - left) / r, (by1 - top) / r)
        out.append((float(scores[i]), box, logit, (cx0, cy0, cx1, cy1)))
    return out


def split_contacts(ids, crop, band):
    """Re-draw the boundary where two beans touch. The model's mask grid is coarse
    (about 8 px at working resolution), so near every bean-to-bean contact the pixels within
    `band` px are released and a watershed on the photo itself hands them back to the bean
    (or paper gap) whose colour edge they belong to. Pixels away from contacts are unchanged."""
    k3 = np.ones((3, 3), np.uint8)
    mx = cv2.dilate(ids, k3)
    mn = cv2.erode(np.where(ids == 0, 65535, ids).astype(np.uint16), k3)
    con = (ids > 0) & (((mx != ids) & (mx > 0)) | ((mn != ids) & (mn != 65535)))
    if not con.any():
        return ids
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    rel = (cv2.dilate(con.astype(np.uint8), ke) > 0) & (ids > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(rel.astype(np.uint8), connectivity=8)
    out = ids.copy()
    H, W = ids.shape
    paper_lab = int(ids.max()) + 1
    pad = 3
    for c in range(1, n):
        x, y, w, h = st[c, :4]
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        sub = ids[y0:y1, x0:x1]
        unk = lab[y0:y1, x0:x1] == c
        mk = np.where(sub > 0, sub, paper_lab).astype(np.int32)
        mk[unk] = 0
        cv2.watershed(np.ascontiguousarray(crop[y0:y1, x0:x1, ::-1]), mk)
        mk[mk == -1] = sub[mk == -1]
        mk[mk == paper_lab] = 0
        o = out[y0:y1, x0:x1]
        o[unk] = mk[unk].astype(np.uint16)
    return out


def refine_instances(dets, crop):
    """Crop-space instance map: YOLO decides WHICH bean each pixel belongs to (highest mask
    logit), the colour rule decides WHERE the bean/paper edge is (a shadow has the paper's
    colour; a bean has its own colour or is very dark). Each bean is cleaned to one solid
    piece (largest component, holes filled) so polygon and mask always agree.
    Returns (uint16 id map of the crop, list of (x0, y0, x1, y1) boxes in crop pixels)."""
    Hc, Wc = crop.shape[:2]
    best = np.full((Hc, Wc), -50.0, np.float32)
    bid = np.zeros((Hc, Wc), np.uint16)
    model_area = []
    for k, (conf, box, logit, (cx0, cy0, cx1, cy1)) in enumerate(dets, 1):
        X0, Y0, X1, Y1 = int(round(cx0)), int(round(cy0)), int(round(cx1)), int(round(cy1))
        if X1 - X0 < 3 or Y1 - Y0 < 3:
            model_area.append(0); continue
        L = cv2.resize(logit.astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
        m = BOX_MARGIN                                   # allow a few pixels beyond the box
        bx0, by0 = int(box[0]) - X0 - m, int(box[1]) - Y0 - m
        bx1, by1 = int(np.ceil(box[2])) - X0 + m, int(np.ceil(box[3])) - Y0 + m
        clip = np.full(L.shape, -50.0, np.float32)
        clip[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = 0
        L = L + clip
        xa, ya, xb, yb = max(0, X0), max(0, Y0), min(Wc, X1), min(Hc, Y1)
        if xb <= xa or yb <= ya:
            model_area.append(0); continue
        L = L[ya - Y0:yb - Y0, xa - X0:xb - X0]
        model_area.append(int((L > 0).sum()))
        bb, ii = best[ya:yb, xa:xb], bid[ya:yb, xa:xb]
        upd = L > bb
        bb[upd] = L[upd]; ii[upd] = k
    # colour rule: bean vs paper
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    far = best[::4, ::4] < -6
    sub = lab[::4, ::4][far]
    paper = np.median(sub, axis=0).astype(np.float32) if len(sub) > 100 else np.array([215, 128, 128], np.float32)
    # (a - pa)^2 + (b - pb)^2 > T^2 as a 256x256 table over the 8-bit a, b values: same float32 maths, done once
    A = np.arange(256, dtype=np.float32) - paper[1]
    B = np.arange(256, dtype=np.float32) - paper[2]
    chroma = ((A[:, None] * A[:, None] + B[None, :] * B[None, :]) > CHROMA_T ** 2).ravel()
    dark = cv2.LUT(np.ascontiguousarray(lab[..., 0]), (np.arange(256) < DARK_T * paper[0]).astype(np.uint8)).view(bool)
    fg = chroma[(lab[..., 1].astype(np.uint16) << 8) | lab[..., 2]] | dark
    keep = (fg & (best > -FG_LOGIT)) | (best > SURE_LOGIT) if REFINE else (best > 0)
    ids0 = np.where(keep, bid, 0).astype(np.uint16)
    # per-bean clean-up: largest piece, fill holes, drop beans that are mostly hidden
    out = np.zeros_like(ids0)
    boxes, feats = [], []
    n = 0
    for k, (conf, box, logit, (cx0, cy0, cx1, cy1)) in enumerate(dets, 1):
        xa, ya = max(0, int(cx0) - 4), max(0, int(cy0) - 4)
        xb, yb = min(Wc, int(np.ceil(cx1)) + 4), min(Hc, int(np.ceil(cy1)) + 4)
        if xb <= xa or yb <= ya:
            continue
        mk = (ids0[ya:yb, xa:xb] == k).astype(np.uint8)
        if mk.sum() < MIN_PIX:
            continue
        nc, lb, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=4)
        if nc > 2:
            mk = (lb == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mk)
        cv2.drawContours(filled, cs, -1, 1, -1)
        region = out[ya:yb, xa:xb]
        free = (filled > 0) & (region == 0) & ((ids0[ya:yb, xa:xb] == k) | (ids0[ya:yb, xa:xb] == 0))
        area = int(free.sum())
        if area < MIN_PIX or area < MIN_VISIBLE * max(1, model_area[k - 1]):
            continue
        n += 1
        region[free] = n
        ys, xs = np.nonzero(free)
        boxes.append((xa + xs.min(), ya + ys.min(), xa + xs.max() + 1, ya + ys.max() + 1))
        feats.append((conf, area, model_area[k - 1]))
    if WS_BAND and n > 1:
        out = split_contacts(out, crop, WS_BAND)
        # clean up again: one solid piece per bean, renumbered 1..N
        out2 = np.zeros_like(out); boxes2, feats2 = [], []; m = 0
        for j, (x0, y0, x1, y1) in enumerate(boxes, 1):
            xa, ya = max(0, x0 - WS_BAND - 2), max(0, y0 - WS_BAND - 2)
            xb, yb = min(Wc, x1 + WS_BAND + 2), min(Hc, y1 + WS_BAND + 2)
            mk = (out[ya:yb, xa:xb] == j).astype(np.uint8)
            if mk.sum() < MIN_PIX:
                continue
            nc, lb, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=4)
            if nc > 2:
                mk = (lb == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled = np.zeros_like(mk)
            cv2.drawContours(filled, cs, -1, 1, -1)
            region = out2[ya:yb, xa:xb]
            free = (filled > 0) & (region == 0)
            area = int(free.sum())
            if area < MIN_PIX:
                continue
            m += 1
            region[free] = m
            ys, xs = np.nonzero(free)
            boxes2.append((xa + xs.min(), ya + ys.min(), xa + xs.max() + 1, ya + ys.max() + 1))
            feats2.append((feats[j - 1][0], area, feats[j - 1][2]))
        out, boxes, feats = out2, boxes2, feats2
    # quality features per kept bean: paper-border fraction, solidity, size vs median
    med = np.median([f[1] for f in feats]) if feats else 1
    quality = []
    if not ({"paper", "solidity"} & set(F1_FILTER or {})):     # the rule in use needs only cheap features
        for conf, area, marea in feats:
            quality.append({"conf": conf, "size": area / max(1.0, med), "keep_frac": area / max(1, marea)})
        QUALITY_LOG.clear(); QUALITY_LOG.extend(quality)
        return out, boxes
    for j, (x0, y0, x1, y1) in enumerate(boxes, 1):
        xa, ya, xb, yb = max(0, x0 - 2), max(0, y0 - 2), min(Wc, x1 + 2), min(Hc, y1 + 2)
        reg = out[ya:yb, xa:xb]
        mk = (reg == j).astype(np.uint8)
        ring = (cv2.dilate(mk, np.ones((3, 3), np.uint8)) > 0) & (mk == 0)
        paper = float((reg[ring] == 0).mean()) if ring.any() else 1.0
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))) if cs else 1
        conf, area, marea = feats[j - 1]
        quality.append({"conf": conf, "paper": paper, "solidity": area / max(1.0, hull),
                        "size": area / max(1.0, med), "keep_frac": area / max(1, marea)})
    QUALITY_LOG.clear(); QUALITY_LOG.extend(quality)
    return out, boxes


def apply_f1_filter(crop_ids, crop_boxes, quality):
    """Keep only beans passing F1_FILTER; renumber ids 1..N."""
    if not F1_FILTER or not crop_boxes:
        return crop_ids, crop_boxes
    keep = [j for j, q in enumerate(quality) if all(q[k] >= v for k, v in F1_FILTER.items())]
    if len(keep) == len(crop_boxes):                  # nothing dropped (the usual case): ids are already 1..N
        return crop_ids, crop_boxes
    lut = np.zeros(len(crop_boxes) + 1, np.uint16)
    for new, j in enumerate(keep, 1):
        lut[j + 1] = new
    return lut[crop_ids], [crop_boxes[j] for j in keep]


def to_full(crop_ids, crop_boxes, crop_off, half_shape, full_wh):
    """Paste the crop id map into a full-size uint16 map (nearest upsampling)."""
    W, H = full_wh
    fx, fy = W / half_shape[1], H / half_shape[0]
    ox, oy = crop_off
    Hc, Wc = crop_ids.shape
    X0, Y0 = int(round(ox * fx)), int(round(oy * fy))
    X1, Y1 = min(W, int(round((ox + Wc) * fx))), min(H, int(round((oy + Hc) * fy)))
    ids = np.zeros((H, W), np.uint16)
    if crop_boxes:
        ids[Y0:Y1, X0:X1] = cv2.resize(crop_ids, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
    boxes = [(max(0, int(X0 + x0 * fx) - 2), max(0, int(Y0 + y0 * fy) - 2),
              min(W, int(X0 + x1 * fx) + 2), min(H, int(Y0 + y1 * fy) + 2)) for x0, y0, x1, y1 in crop_boxes]
    return ids, boxes


# Polygon simplification in pixels of the full photo. 0 = the exact mask outline (only points on straight runs
# dropped: same filled shape). The scored latency is the latency_ms we publish (QA 19/11), not the upload, so the
# exact outline costs nothing; 1.0 would make the answer ~1/3 the size for -0.001 mean IoU.
POLY_EPS = float(os.environ.get("TGR_POLY_EPS", "0"))


def polygons_from_ids(ids, boxes):
    polys = []
    for k, (x0, y0, x1, y1) in enumerate(boxes, 1):
        m = (ids[y0:y1, x0:x1] == k).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            polys.append([]); continue
        c = max(cs, key=cv2.contourArea)
        if POLY_EPS > 0:
            d = cv2.approxPolyDP(c, POLY_EPS, True)
            c = d if len(d) >= 3 else cv2.approxPolyDP(c, 0.0, True)
        else:
            c = cv2.approxPolyDP(c, 0.0, True)                  # drops only points on straight runs: same shape
        polys.append((c.reshape(-1, 2) + [x0, y0]).tolist())
    return polys


# ---------------------------------------------------------------- classifier per bean
def median_u8(img, mask):
    """np.median(img[mask], axis=0) for an 8-bit HxWxC image, exactly, from per-channel histograms (~2x faster)."""
    m = mask.view(np.uint8) if mask.dtype == bool else mask
    n = int(np.count_nonzero(m))
    src = np.ascontiguousarray(img)
    out = np.empty(img.shape[2], np.float64)
    for c in range(img.shape[2]):
        cs = np.cumsum(cv2.calcHist([src], [c], np.ascontiguousarray(m), [256], [0, 256]).ravel())
        lo = int(np.searchsorted(cs, (n - 1) // 2, side="right"))      # sorted value at index (n-1)//2
        hi = int(np.searchsorted(cs, n // 2, side="right"))            # sorted value at index n//2
        out[c] = (lo + hi) / 2
    return out


def _class_probs(cls, crops):
    x = ((np.stack(crops).astype(np.float32) / 255 - 0.5) / 0.25).transpose(0, 3, 1, 2)
    return cls.forward(np.ascontiguousarray(x))


def classify(cls, half, ids, boxes, full_wh, half_ids=None):
    """One label per bean. Up to CLS_MAX_BEANS beans (largest first) are classified in ONE batch; the 12
    largest set the photo majority, used only for beans beyond CLS_MAX_BEANS. Crops are cut straight from the
    photo (only crops at its edge are padded), touching beans painted out, white-balanced to the paper."""
    if cls is None or not boxes:
        return ["good"] * len(boxes)
    W, H = full_wh
    f = half.shape[1] / W                         # full -> half-size pixels
    win = int(round(CLS_WIN * half.shape[1] / CLS_WORK_W))   # same physical window as Day 1
    Hh, Wh = half.shape[:2]
    ih = half_ids if half_ids is not None else cv2.resize(ids, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST)
    bgm = ih[::4, ::4] == 0
    paper = median_u8(half[::4, ::4], bgm).astype(np.uint8) if bgm.sum() > 100 else np.array([215, 212, 205], np.uint8)
    cent, areas = [], []
    for k, (x0, y0, x1, y1) in enumerate(boxes, 1):            # area and centre of each bean, full-size mask
        M = cv2.moments((ids[y0:y1, x0:x1] == k).view(np.uint8), True)
        areas.append(M["m00"])
        cent.append(((M["m10"] / max(M["m00"], 1) + x0) * f, (M["m01"] / max(M["m00"], 1) + y0) * f))
    order = list(np.argsort(areas)[::-1])

    def cut(img, X0, Y0, border):
        """img[Y0:Y0+win, X0:X0+win], padded like cv2.copyMakeBorder(img, ..., border) where it leaves the image."""
        X1, Y1 = X0 + win, Y0 + win
        if X0 >= 0 and Y0 >= 0 and X1 <= Wh and Y1 <= Hh:
            return img[Y0:Y1, X0:X1]
        part = img[max(0, Y0):min(Hh, Y1), max(0, X0):min(Wh, X1)]
        return cv2.copyMakeBorder(part, max(0, -Y0), max(0, Y1 - Hh), max(0, -X0), max(0, X1 - Wh), border, value=0)

    def crop_of(i):
        cx, cy = cent[i]
        X, Y = int(cx) - win // 2, int(cy) - win // 2
        c = cut(half, X, Y, cv2.BORDER_REPLICATE).copy()
        w = cut(ih, X, Y, cv2.BORDER_CONSTANT)
        c[(w != 0) & (w != i + 1)] = paper              # paint touching beans out (as in training)
        r = cv2.resize(c, (CLS_CROP, CLS_CROP), interpolation=cv2.INTER_AREA)
        if CLS_WB:                                       # white balance: paper -> neutral grey
            r = np.clip(r.astype(np.float32) * (CLS_REF / np.maximum(paper.astype(np.float32), 1)), 0, 255).astype(np.uint8)
        return r

    sel = order[:CLS_MAX_BEANS]
    P = _class_probs(cls, [crop_of(i) for i in sel])
    first = P[:min(12, len(sel))]
    votes = np.bincount(first.argmax(1), minlength=len(CLASSES))
    tied = np.flatnonzero(votes == votes.max())
    major = CLASSES[int(tied[first.sum(0)[tied].argmax()])]
    labels = [major] * len(boxes)
    for i, p in zip(sel, P.argmax(1)):
        labels[i] = CLASSES[int(p)]
    return labels


# ---------------------------------------------------------------- main
def predict(src, models, pre=None):
    """Masks, polygons and per-bean classes for one photo (file path or file-like object). Writes nothing.
    pre: (half, full_wh) already decoded during the download (StreamDecoder), else the photo is decoded here."""
    sess, cls = models
    half, full_wh = pre if pre is not None else load_image(src)
    x0, y0, x1, y1 = bean_region(half)
    crop = half[y0:y1, x0:x1]
    dets = run_seg(sess, crop)
    crop_ids, crop_boxes = refine_instances(dets, crop)
    crop_ids, crop_boxes = apply_f1_filter(crop_ids, crop_boxes, list(QUALITY_LOG))
    ids, boxes = to_full(crop_ids, crop_boxes, (x0, y0), half.shape, full_wh)
    polys = polygons_from_ids(ids, boxes)
    half_ids = np.zeros(half.shape[:2], np.uint16)          # the same bean ids at working resolution
    half_ids[y0:y0 + crop_ids.shape[0], x0:x0 + crop_ids.shape[1]] = crop_ids
    classes = classify(cls, half, ids, boxes, full_wh, half_ids)
    for k, p in enumerate(polys, 1):
        if len(p) < 3:
            raise ValueError(f"bean {k} has a polygon with fewer than 3 points")
    return {"half": half, "full_wh": full_wh, "ids": ids, "boxes": boxes, "polys": polys, "classes": classes}


def result_json(name, pred):
    """predictions.json / API entry for one photo."""
    objects = [{"id": k, "class": c, "polygon": p} for k, (c, p) in enumerate(zip(pred["classes"], pred["polys"]), 1)]
    return {"image_name": name, "objects": objects}


def write_outputs(name, pred, out_dir):
    """seg_mask PNG + class-coloured overlay for one photo."""
    half, full_wh, ids, boxes, classes = pred["half"], pred["full_wh"], pred["ids"], pred["boxes"], pred["classes"]
    stem = os.path.splitext(name)[0]
    Image.fromarray(ids.astype(np.uint16)).save(os.path.join(out_dir, "seg_mask", stem + ".png"), compress_level=1)
    # quarter-size overlay, coloured by class, with each bean's id and class
    vis = cv2.resize(half, (half.shape[1] // 2, half.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    small_ids = cv2.resize(ids, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    colors = np.zeros((len(boxes) + 1, 3), np.uint8)
    for k, c in enumerate(classes, 1):
        colors[k] = CLASS_RGB.get(c, (255, 255, 255))
    over = colors[small_ids]
    vis = np.where(small_ids[..., None] > 0, (0.55 * vis + 0.45 * over).astype(np.uint8), vis)
    fs = vis.shape[1] / full_wh[0]
    for k, ((bx0, by0, bx1, by1), c) in enumerate(zip(boxes, classes), 1):
        cx, cy = int((bx0 + bx1) / 2 * fs), int((by0 + by1) / 2 * fs)
        cv2.putText(vis, f"{k}:{c[:3]}", (cx - 30, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, f"{k}:{c[:3]}", (cx - 30, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, f"{len(boxes)} beans", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.imwrite(os.path.join(out_dir, "vis", stem + ".jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def process(path, models, out_dir):
    pred = predict(path, models)
    name = os.path.basename(path)
    write_outputs(name, pred, out_dir)
    return result_json(name, pred)


def mqtt_message(result, seconds):
    """Per-photo message for the Agent (slide 7): counts of all 7 classes, defect ratio, latency."""
    counts = {c: 0 for c in ["good", "shell", "black", "husk", "broken", "parchment", "immature"]}
    for o in result["objects"]:
        counts[o["class"]] += 1
    total = sum(counts.values())
    defect = sum(counts[c] for c in DEFECTS)
    return {"image": result["image_name"], "latency_ms": int(round(seconds * 1000)), "counts": counts,
            "defect_ratio": round(defect / total, 4) if total else 0.0}


def warm_up(models):
    """Run everything once so the first timed photo is not slowed by cold start-up."""
    run_seg(models[0], np.full((IMGSZ, IMGSZ, 3), 200, np.uint8))
    if models[1] is not None:
        models[1].pick_fastest()
    Image.fromarray(np.zeros((8, 8), np.uint16)).save(io.BytesIO(), format="PNG", compress_level=1)
    cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    # the whole pipeline on a real photo (weights/warmup.jpg, a Day 3 training photo), so the first timed photo is
    # not slower than the rest: in the 19/11 rehearsals photo 1 took 50-70 ms longer
    p = os.path.join(getattr(models[0], "weights_dir", "") or "", "warmup.jpg")
    if os.path.isfile(p):
        for _ in range(2):
            try:
                result_json("warmup.jpg", predict(p, models))
            except Exception:                                   # noqa: BLE001 — warm-up must never stop a run
                break


def say(*args, **kw):
    """Print, but never let a closed or broken terminal stop the run."""
    try:
        print(*args, **kw)
    except (OSError, ValueError):
        pass


def main():
    ap = argparse.ArgumentParser(description="TGR-21 Day 3: mask, outline and class of every coffee bean")
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True, help="weights folder (seg_ncnn_model/ or seg.onnx, bean_cls.onnx, bean_cls_ncnn/, warmup.jpg)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.isfile(a.input):
        images = [a.input]
    elif os.path.isdir(a.input):
        images = sorted(os.path.join(a.input, f) for f in os.listdir(a.input)
                        if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(a.input, f)))
    else:
        fail(f"input not found: {a.input}")
    if not images:
        fail(f"no images in {a.input}")
    stems = [os.path.splitext(os.path.basename(p))[0].lower() for p in images]
    if len(set(stems)) != len(stems):
        fail("input file names (without extension) must be unique, or masks would overwrite each other")
    out = os.path.abspath(a.out)
    in_dir = os.path.abspath(a.input if os.path.isdir(a.input) else os.path.dirname(a.input))
    if out == in_dir:
        fail("--out must be different from the input folder")
    models = load_models(a.weights)
    for d in ("seg_mask", "vis", "mqtt"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    warm_up(models)
    say(f"runtime: {models[0].kind}", flush=True)
    results, times = [], []
    for p in images:
        t = time.perf_counter()
        try:
            results.append(process(p, models, out))
        except Exception as e:
            fail(f"could not process {p}: {e}")
        times.append(time.perf_counter() - t)
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(out, "mqtt", stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(mqtt_message(results[-1], times[-1]), f)
        cnt = {c: 0 for c in CLASSES}
        for o in results[-1]["objects"]:
            cnt[o["class"]] += 1
        kinds = "  ".join(f"{c} {n}" for c, n in cnt.items() if n) or "no beans"
        say(f"{os.path.basename(p)}: {len(results[-1]['objects'])} beans, {times[-1]:.2f} s   | {kinds}", flush=True)
    with open(os.path.join(out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)
    # summary table: beans of each defect type per photo
    w = max(10, max(len(r["image_name"]) for r in results))
    head = f"{'photo':<{w}} {'beans':>5} " + " ".join(f"{c:>9}" for c in CLASSES) + f" {'time':>6}"
    say("\n" + head + "\n" + "-" * len(head))
    tot = {c: 0 for c in CLASSES}
    for r, t in zip(results, times):
        cnt = {c: 0 for c in CLASSES}
        for o in r["objects"]:
            cnt[o["class"]] += 1; tot[o["class"]] += 1
        say(f"{r['image_name']:<{w}} {len(r['objects']):>5} " + " ".join(f"{cnt[c]:>9}" for c in CLASSES) + f" {t:>5.2f}s")
    say("-" * len(head))
    say(f"{'total':<{w}} {sum(tot.values()):>5} " + " ".join(f"{tot[c]:>9}" for c in CLASSES) + f" {'':>6}\n")
    say(f"done: {len(images)} images, median {np.median(times):.2f} s/image -> {out}")


if __name__ == "__main__":
    main()
