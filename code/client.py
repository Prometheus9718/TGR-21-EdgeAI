"""TGR-21 Day 3 client for the TESA API (API_DOCS, 19/11).

Per photo:  GET /api/v1/image (the JPEG is decoded while it downloads)  ->  inference in this process
            -> POST /api/v1/submit {img_id, predictions}
            -> publish {image, latency_ms, counts, defect_ratio} to MQTT (background thread, never delays the next photo)
            -> save seg_mask/, vis/, mqtt/, predictions.json and the photo (after the POST)
latency_ms = time from the photo being fully received to the answer being ready, on this machine. It is the
latency that is scored (QA 19/11); the server's own time (photo sent -> JSON received) is a cross-check.
Models are loaded and run once on weights/warmup.jpg, and MQTT is connected, before the first photo.

Credentials: team_config.json next to this file (or environment variables of the same names).
It is never put into a zip.

  python client.py --check                      # GET /api/v1/stats + MQTT login test (no photo used, nothing published)
  python client.py --mode train --count 16      # rehearsal with training photos
  python client.py --mode test                  # real test: asks for a typed YES, then the remaining quota, one by one
  python client.py --dry-run --input <folder>   # no network: run the same loop on local photos
  python client.py --resend-mqtt                # publish outputs/mqtt/*.json again if MQTT failed
  python client.py --make-zip                   # TGR-21_Day3.zip: the repo without outputs/ and team_config.json
"""
import argparse, io, json, os, queue, re, ssl, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import infer  # noqa: E402

DEFAULTS = {"HTTP_URL": "", "TEAM_PASSWORD": "", "TEAM_ID": "TGR-21", "MQTT_KEY": "",
            "MQTT_HOST": "tesa-score.c333m.site", "MQTT_PORT": 443, "MQTT_PATH": "/mqtt/", "MQTT_TLS": True,
            "MQTT_CLIENT_ID": "TGR-21:device-1", "TRACK": "EdgeAI", "EVENT": "1", "MQTT_PROTOCOL": "5"}
LOG_PATH = os.path.join(HERE, "logs", time.strftime("client_%Y%m%d_%H%M%S.log"))


def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    try:
        print(line, flush=True)
    except (OSError, ValueError):
        pass
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_config(path):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    for k in cfg:                                  # environment variables win
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    cfg["HTTP_URL"] = str(cfg["HTTP_URL"]).rstrip("/")
    return cfg


# ------------------------------------------------------------------ MQTT (background publisher)
class Publisher:
    def __init__(self, cfg, enabled=True):
        self.cfg, self.enabled, self.q = cfg, enabled, queue.Queue()
        self.sent, self.failed, self.connected = 0, [], threading.Event()
        self.client = None
        if enabled:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _connect(self):
        import paho.mqtt.client as mqtt
        proto = mqtt.MQTTv5 if str(self.cfg.get("MQTT_PROTOCOL", "5")) == "5" else mqtt.MQTTv311
        c = cfg_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.cfg["MQTT_CLIENT_ID"],
                                     transport="websockets", protocol=proto)
        c.username_pw_set(self.cfg["TEAM_ID"], self.cfg["MQTT_KEY"])
        c.ws_set_options(path=self.cfg["MQTT_PATH"])
        if str(self.cfg["MQTT_TLS"]).lower() not in ("false", "0", "no"):
            c.tls_set_context(ssl.create_default_context())

        def on_connect(cl, userdata, flags, reason, props=None):
            if not reason.is_failure:
                self.connected.set(); log(f"MQTT connected to {self.cfg['MQTT_HOST']} as {self.cfg['TEAM_ID']}")
            else:
                log(f"MQTT connect refused: {reason}")

        def on_disconnect(cl, userdata, flags, reason, props=None):
            self.connected.clear(); log(f"MQTT disconnected: {reason}")
        c.on_connect, c.on_disconnect = on_connect, on_disconnect
        c.reconnect_delay_set(1, 10)
        if str(self.cfg.get("MQTT_TRANSPORT", "websockets")) == "tcp":    # local testing only
            c._transport = "tcp"
        c.connect_async(self.cfg["MQTT_HOST"], int(self.cfg["MQTT_PORT"]), keepalive=30)
        c.loop_start()
        self.client = cfg_client

    def topic(self):
        return f"/event/{self.cfg['MQTT_KEY']}/{self.cfg['TRACK']}/{self.cfg['EVENT']}"

    def _run(self):
        try:
            self._connect()
        except Exception as e:                                    # noqa: BLE001
            log(f"MQTT setup failed: {e}")
        while True:
            msg = self.q.get()
            if msg is None:
                break
            ok = False
            for attempt in range(1, 6):
                if not self.connected.wait(timeout=10):
                    log(f"MQTT not connected yet (attempt {attempt}/5) for {msg['image']}")
                    continue
                try:
                    info = self.client.publish(self.topic(), json.dumps(msg, separators=(",", ":")), qos=1)
                    info.wait_for_publish(timeout=10)
                    if info.is_published():
                        ok = True; break
                except Exception as e:                            # noqa: BLE001
                    log(f"MQTT publish error (attempt {attempt}/5): {e}")
                time.sleep(1)
            if ok:
                self.sent += 1; log(f"MQTT published {msg['image']}: {msg['counts']} defect_ratio {msg['defect_ratio']}")
            else:
                self.failed.append(msg); log(f"MQTT FAILED for {msg['image']} — resend later with: python client.py --resend-mqtt")
            self.q.task_done()

    def publish(self, msg):
        if self.enabled:
            self.q.put(msg)
        else:
            log(f"[mqtt off] would publish: {json.dumps(msg)}")

    def close(self, timeout=60):
        if not self.enabled:
            return
        t0 = time.time()
        while self.q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.2)
        self.q.put(None)
        if self.client is not None:
            time.sleep(0.5); self.client.loop_stop(); self.client.disconnect()


# ------------------------------------------------------------------ HTTP
def session(cfg):
    import requests
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {cfg['TEAM_PASSWORD']}"})
    return s


def get_stats(s, cfg):
    r = s.get(cfg["HTTP_URL"] + "/api/v1/stats", timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"/stats -> {r.status_code} {r.text[:200]}")
    return r.json()


def get_image(s, cfg, mode, stream_decode=True):
    """GET one photo. The JPEG is decoded piece by piece while it downloads (r.dec), because the server's
    clock is already running during the download."""
    params = {"mode": mode} if mode else None
    for attempt in range(1, 6):
        t0 = time.perf_counter()
        try:
            r = s.get(cfg["HTTP_URL"] + "/api/v1/image", params=params, timeout=30, stream=True)
        except Exception as e:                                    # noqa: BLE001 — network hiccup: retry
            log(f"GET image error (attempt {attempt}/5): {e}"); time.sleep(2); continue
        if r.status_code == 200:
            dec = infer.StreamDecoder() if stream_decode else None
            buf = bytearray()
            try:
                for chunk in r.iter_content(chunk_size=16384):
                    if dec is not None:
                        dec.feed(chunk)
                    else:
                        buf += chunk
            except Exception as e:                                # noqa: BLE001 — connection broke mid-photo
                log(f"photo download broke (attempt {attempt}/5, img_id {r.headers.get('X-Image-ID')}): {e}")
                r.close(); time.sleep(1); continue
            r.dl_ms = (time.perf_counter() - t0) * 1000           # request sent -> whole photo received
            r.photo = bytes(dec.buf) if dec is not None else bytes(buf)
            r.dec = dec
            return r
        if r.status_code in (401, 403):
            log(f"GET image -> {r.status_code} {r.text[:200]}  ({'wrong team password' if r.status_code == 401 else 'quota used up or mode closed'})")
            return None
        log(f"GET image -> {r.status_code} {r.text[:120]} (attempt {attempt}/5)"); time.sleep(3)
    return None


def submit(s, cfg, img_id, result):
    body = json.dumps({"img_id": img_id, "predictions": result}, separators=(",", ":"))
    info = {"kb": len(body) / 1024}
    for attempt in range(1, 4):
        t0 = time.perf_counter()
        try:
            r = s.post(cfg["HTTP_URL"] + "/api/v1/submit", data=body, headers={"Content-Type": "application/json"}, timeout=30)
        except Exception as e:                                    # noqa: BLE001
            log(f"POST error (attempt {attempt}/3): {e}"); time.sleep(1); continue
        info["ms"] = (time.perf_counter() - t0) * 1000            # upload + server reply
        if r.status_code == 200:
            try:
                return r.status_code, r.json(), info
            except ValueError:
                return r.status_code, {"message": r.text[:200]}, info
        if r.status_code in (408, 409, 401, 403):                 # too late / already submitted / auth / closed: do not retry
            return r.status_code, {"message": r.text[:200]}, info
        log(f"POST -> {r.status_code} {r.text[:120]} (attempt {attempt}/3)"); time.sleep(1)
    return None, {}, info


def make_zip(out_dir, name="TGR-21_Day3", with_outputs=False):
    """Submission zip: README, requirements, train.py, infer.py, client.py, helpers, weights/ — never team_config.json.
    outputs/ only with --with-outputs (QA 19/11: the zip has no outputs and is submitted before 12:00, not over HTTP)."""
    import zipfile
    path = os.path.join(HERE, name + ".zip")
    skip_dirs = {"__pycache__", "logs", "test_data", "outputs"}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith((".", "rehearsal_")) and os.path.join(root, d) != os.path.abspath(out_dir)]
            for f in files:
                full = os.path.join(root, f)
                if f == "team_config.json" or f.endswith(".zip") or f.startswith("."):
                    continue
                z.write(full, f"{name}/" + os.path.relpath(full, HERE))
        for root, _, files in (os.walk(out_dir) if with_outputs else []):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, f"{name}/outputs/" + os.path.relpath(full, out_dir))
    log(f"zip ready: {path} ({os.path.getsize(path) / 1e6:.1f} MB) — team_config.json not included")


# ------------------------------------------------------------------ main loop
def safe_name(name, k):
    name = os.path.basename(str(name or "")).strip() or f"image_{k:02d}.jpg"
    return re.sub(r"[^\w.\-]", "_", name)


VERSION = "final"


def main():
    ap = argparse.ArgumentParser(description="TGR-21 Day 3 client: TESA HTTP API + MQTT")
    ap.add_argument("--config", default=os.path.join(HERE, "team_config.json"))
    ap.add_argument("--mode", choices=["train", "test"], help="REQUIRED for a real run: train (rehearsal, unlimited) or test (uses the 5-photo quota)")
    ap.add_argument("--count", type=int, help="photos to process (test mode default: remaining quota from /stats)")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--input", default=os.path.join(HERE, "test_data"), help="downloaded photos are saved here (dry-run: read from here)")
    ap.add_argument("--check", action="store_true", help="test HTTP login (/stats) and MQTT login; no photo is requested")
    ap.add_argument("--dry-run", action="store_true", help="no network: run the loop on photos already in --input")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--resend-mqtt", action="store_true", help="publish every outputs/mqtt/*.json again (if MQTT failed during the run)")
    ap.add_argument("--make-zip", action="store_true", help="build TGR-21_Day3.zip (repo + outputs, no passwords) and exit")
    ap.add_argument("--with-outputs", action="store_true", help="--make-zip: also put outputs/ in the zip")
    ap.add_argument("--yes", action="store_true", help="skip the typed YES that --mode test asks for (e.g. when run with nohup)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.make_zip:                                # building the zip needs no server or MQTT settings
        make_zip(a.out, with_outputs=a.with_outputs); return
    net = not a.dry_run
    if net and not (a.check or a.resend_mqtt or a.make_zip) and a.mode is None:
        sys.exit("Say which photos to use: --mode train (rehearsal, unlimited) or --mode test (real test, 5-photo quota).")
    if net and (not cfg["HTTP_URL"] or not cfg["TEAM_PASSWORD"]) and not a.resend_mqtt:
        sys.exit(f"Fill HTTP_URL and TEAM_PASSWORD in {a.config} first.")
    if (net or a.resend_mqtt) and not a.no_mqtt and not cfg["MQTT_KEY"]:
        sys.exit(f"Fill MQTT_KEY in {a.config} first (or use --no-mqtt).")


    if a.check:
        s = session(cfg)
        try:
            log("HTTP /stats:", json.dumps(get_stats(s, cfg)))
        except Exception as e:                                    # noqa: BLE001
            log("HTTP check FAILED:", e)
        if not a.no_mqtt:
            p = Publisher(cfg)
            log("MQTT login:", "OK" if p.connected.wait(timeout=15) else "FAILED (no connection within 15 s)")
            p.close(timeout=1)
        return

    if a.resend_mqtt:
        p = Publisher(cfg)
        for f in sorted(os.listdir(os.path.join(a.out, "mqtt"))):
            with open(os.path.join(a.out, "mqtt", f), encoding="utf-8") as fh:
                p.publish(json.load(fh))
        p.close(); log(f"resend done: {p.sent} published, {len(p.failed)} failed"); return

    if net and a.mode == "test" and not a.yes:    # the 5 real attempts cannot be undone: make starting them deliberate
        try:
            left = get_stats(session(cfg), cfg).get("test_mode", {}).get("remaining_attempts", "?")
        except Exception as e:                    # noqa: BLE001
            left = f"? (could not read /stats: {e})"
        if not sys.stdin.isatty():
            sys.exit("--mode test asks for a typed YES; add --yes if you run it without a terminal (nohup).")
        print(f"\nREAL TEST: this uses the real test attempts ({left} left). It cannot be undone.")
        if input("Type YES to start: ").strip() != "YES":
            sys.exit("Not started.")
    log(f"TGR-21 Day 3 client {VERSION}, mode {a.mode or 'dry-run'}")
    pub = Publisher(cfg, enabled=net and not a.no_mqtt)   # connects in the background while the models load
    log("loading models ..."); t0 = time.time()
    models = infer.load_models(a.weights); infer.warm_up(models)
    log(f"models ready ({time.time() - t0:.1f} s), segmenter {models[0].kind}, classifier {models[1].kind if models[1] is not None else '-'} (the faster here)")
    stream_ok = infer.stream_self_test()
    log("decode during download:", "on (self-test passed)" if stream_ok else "OFF (self-test failed: photos are decoded after download)")
    for d in ("seg_mask", "vis", "mqtt"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    os.makedirs(a.input, exist_ok=True)
    if pub.enabled and not pub.connected.wait(timeout=10):   # its encryption handshake must not compete with photo 1
        log("MQTT not connected yet: starting anyway (messages wait in a queue and are retried)")
    s = session(cfg) if net else None
    sp = session(cfg) if net else None            # second connection used only for the POST, opened before each photo

    if net:
        count = a.count
        if count is None:
            try:
                st = get_stats(s, cfg); log("stats:", json.dumps(st))
                count = int(st.get("test_mode", {}).get("remaining_attempts", 0)) if a.mode == "test" else 16
            except Exception as e:                                # noqa: BLE001
                log("could not read /stats:", e); count = 5 if a.mode == "test" else 16
        log(f"processing up to {count} photo(s), mode={a.mode or 'server default'}")
        jobs = range(count)
    else:
        files = sorted(f for f in os.listdir(a.input) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        jobs = files; log(f"dry run on {len(files)} local photo(s)")

    rows = []
    pred_path = os.path.join(a.out, "predictions.json")
    results = {}                                  # merge with an earlier (interrupted) run instead of replacing it
    if os.path.exists(pred_path):
        try:
            with open(pred_path, encoding="utf-8") as fh:
                results = {e["image_name"]: e for e in json.load(fh)}
        except (OSError, ValueError, KeyError):
            results = {}
    for k, job in enumerate(jobs, 1):
        dl_ms = None
        if net:
            try:                                  # open/refresh the POST connection now, before the server starts its clock
                sp.get(cfg["HTTP_URL"] + "/api/v1/stats", timeout=5)
            except Exception:                     # noqa: BLE001
                pass
            r = get_image(s, cfg, a.mode, stream_ok)
            if r is None:
                log("no more photos from the server — stopping"); break
            img_id, name, mode = r.headers.get("X-Image-ID"), safe_name(r.headers.get("X-Image-Name"), k), r.headers.get("X-Mode")
            raw = r.photo; dl_ms = getattr(r, "dl_ms", None); dec = r.dec
            if k == 1:
                hdr = {h.lower(): v for h, v in r.headers.items()}
                log(f"server: {hdr.get('server', '?')} | Connection: {hdr.get('connection', 'keep-alive')} | "
                    f"Cloudflare: {'yes' if 'cf-ray' in hdr else 'no'} | photo type {hdr.get('content-type', '?')}")
        else:
            dec = None
            img_id, name, mode = f"local_{k}", job, "dry-run"
            with open(os.path.join(a.input, job), "rb") as fh:
                raw = fh.read()
        t = time.perf_counter()
        pre = None
        try:
            pre = dec.result() if dec is not None else None       # decoded during the download (None: decode now)
            pred = infer.predict(io.BytesIO(raw), models, pre=pre); result = infer.result_json(name, pred)
        except Exception as e:                                    # noqa: BLE001 — never lose a quota photo: send an empty answer
            log(f"inference FAILED on {name}: {e} — submitting an empty result"); pred, result = None, {"image_name": name, "objects": []}
        latency = time.perf_counter() - t
        status, resp, info = submit(sp, cfg, img_id, result) if net else ("dry-run", {}, {})
        # ---- everything below is after the POST (not timed by the server) and must never stop the run
        msg = infer.mqtt_message(result, latency)
        pub.publish(msg)
        stem = os.path.splitext(name)[0]
        try:
            with open(os.path.join(a.out, "mqtt", stem + ".json"), "w", encoding="utf-8") as fh:
                json.dump(msg, fh)
            if net:
                with open(os.path.join(a.input, name), "wb") as fh:
                    fh.write(raw)
            if pred is not None:
                infer.write_outputs(name, pred, a.out)
            else:                                 # empty answer: empty mask + blank overlay of the photo's size
                from PIL import Image
                import numpy as np
                wh = infer.image_size(raw)
                if wh:
                    W, H = wh
                    Image.fromarray(np.zeros((H, W), np.uint16)).save(os.path.join(a.out, "seg_mask", stem + ".png"))
                    Image.new("RGB", (max(1, W // 4), max(1, H // 4)), (128, 128, 128)).save(os.path.join(a.out, "vis", stem + ".jpg"))
                else:
                    log(f"{name} is not a readable image — no seg_mask/vis written")
        except Exception as e:                    # noqa: BLE001
            log(f"writing outputs for {name} failed: {e} (the answer was already submitted)")
        results[name] = result
        try:
            with open(pred_path, "w", encoding="utf-8") as fh:
                json.dump(list(results.values()), fh)
        except OSError as e:
            log(f"could not write predictions.json: {e}")
        dur = resp.get("duration_seconds") if isinstance(resp, dict) else None
        dur = dur if isinstance(dur, (int, float)) else None
        row = {"name": name, "beans": len(result["objects"]), "mb": len(raw) / 1e6, "dl": dl_ms, "model": latency * 1000, "streamed": pre is not None,
               "kb": info.get("kb"), "up": info.get("ms"), "server": dur, "status": status}
        rows.append(row)
        f0 = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else "-"   # noqa: E731
        log(f"[{k}] {name} ({mode}): {row['beans']} beans | download {row['mb']:.2f} MB {f0(dl_ms, '.0f')} ms | "
            f"model {row['model']:.0f} ms | upload {f0(row['kb'], '.0f')} KB {f0(row['up'], '.0f')} ms | POST {status} | "
            f"server time {f0(dur, '.3f')} s | {msg['counts']}")
    pub.close()
    if rows:
        import statistics
        f0 = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else "-"   # noqa: E731
        log("summary (model = the latency_ms published over MQTT; download/upload measured here; server = its cross-check time):")
        log(f"{'photo':<26} {'beans':>5} {'MB':>5} {'download ms':>11} {'model ms':>8} {'upload KB':>9} {'upload ms':>9} {'server s':>8} {'POST':>5}")
        for r_ in rows:
            log(f"{r_['name'][:26]:<26} {r_['beans']:>5} {r_['mb']:>5.2f} {f0(r_['dl'], '.0f'):>11} {r_['model']:>8.0f} "
                f"{f0(r_['kb'], '.0f'):>9} {f0(r_['up'], '.0f'):>9} {f0(r_['server'], '.3f'):>8} {str(r_['status']):>5}")
        med = lambda key: statistics.median([r_[key] for r_ in rows if isinstance(r_[key], (int, float))] or [float("nan")])   # noqa: E731
        log(f"decoded during download: {sum(r_['streamed'] for r_ in rows)}/{len(rows)} photos")
        log(f"MEDIAN  download {med('dl'):.0f} ms | model {med('model'):.0f} ms | upload {med('up'):.0f} ms | server {med('server'):.3f} s"
            + (f" | MQTT published {pub.sent}, failed {len(pub.failed)}" if pub.enabled else ""))


if __name__ == "__main__":
    main()
