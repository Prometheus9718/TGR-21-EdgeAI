# TGR-21 · Edge AI Day 3 (Integration) · coffee beans on a Raspberry Pi 5

Every bean in a mixed pile: its mask, its outline and its class (good, shell, black, husk, broken, parchment,
immature), on the Pi's CPU. `client.py` takes each test photo over HTTP, answers over HTTP and publishes a
summary per photo over MQTT, all from one script.

Day 3 scoring (slides Rev0c + QA 19/11): Segmentation 15 + Classification 15 + Latency 25 (the `latency_ms`
we publish over MQTT, ranked) + MQTT 5.

## Results (measured 19/11)
| | Masks: mean IoU (missed / extra bean = 0) | Beans found | Class right |
|---|---|---|---|
| 16 real Day 3 photos, judges' hand-drawn outlines | 0.967 | 124 / 124, no extras | 97.6% (photos used in training) |
| Day 3 piles, beans the segmenter never saw | 0.919 | 96% (missed beans lie mostly under others) | 90.8% |
| Day 1 piles, beans no model ever saw | 0.919 | 96% | 89.0% |

Raspberry Pi 5 (NCNN): about 190–230 ms per photo from "photo received" to "answer ready" (`latency_ms`);
segmenter ~120 ms, edge refinement ~35 ms, classifier ~23 ms. The first photo is as fast as the rest.

## On the Pi
```sh
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt   # once
```
`team_config.json` (next to the scripts, never zipped): `HTTP_URL`, `TEAM_PASSWORD` (website login),
`TEAM_ID`, `MQTT_KEY` (dashboard), `MQTT_HOST/PORT/PATH/TLS`, `MQTT_CLIENT_ID`, `TRACK`, `EVENT`.

Test day, inside `tmux new -s tgr` (survives SSH drops: `tmux attach -t tgr`):
```sh
python client.py --make-zip         # before 12:00: TGR-21_Day3.zip = this repo without outputs/ and team_config.json
bash preflight.sh                   # on its own line: settings + MQTT event, rehearsal results aside, network, logins
python client.py --mode test        # asks for a typed YES, then answers the remaining test photos one by one
python check_outputs.py test_data outputs
```

| Command | What it does |
|---|---|
| `python client.py --check` | server login (`/api/v1/stats`, shows remaining test attempts) + MQTT login; uses no photo |
| `python client.py --mode train --count 16` | rehearsal with training photos |
| `python client.py --mode test` | the real test (typed YES; `--yes` skips it) |
| `python client.py --dry-run --input <folder>` | the same loop on local photos, no network |
| `python client.py --resend-mqtt` | publish `outputs/mqtt/*.json` again (e.g. after fixing `EVENT`) |
| `python client.py --make-zip [--with-outputs]` | submission zip |
| `python infer.py --input <folder> --weights weights/ --out outputs/` | inference only, same 3 arguments as Day 2 |
| `sudo bash net_tune.sh` | before a run, never during one: Wi-Fi power saving off, faster internet path, CPU at full clock |
| `python profile_stages.py` | time of each pipeline step on the photos in `test_data/` |

## Outputs (`outputs/`)
- `seg_mask/<photo>.png` — full size, uint16, 0 = paper, 1..N = bean id.
- `predictions.json` — per photo `image_name` + `objects` [`id`, `class`, `polygon`]; `id` = value in the mask,
  polygon = exact outline of the same mask in full-photo pixels. This is what `client.py` POSTs.
- `mqtt/<photo>.json` — the message published per photo: `image`, `latency_ms`, `counts` (all 7 classes, zeros
  included), `defect_ratio` = beans not `good` / all beans. Counts come from the same per-bean classes.
- `vis/<photo>.jpg` — overlay coloured by class, with each bean's id and class.

## How it works (`infer.py`, `client.py`)
1. **Decode while downloading**: the JPEG is decoded at half size (libjpeg's own 1/2 scaling) piece by piece as
   it arrives, so almost nothing is left to decode when the last byte lands (identical pixels to a normal decode,
   checked by a self-test at start).
2. **Bean region**: a cheap low-resolution threshold finds where the beans are; only that region is processed.
3. **Segmenter**: YOLO26n-seg, one class "bean", run at 640 (trained at 896; see `train.py`).
4. **Edges**: the model decides which bean a pixel belongs to; a colour rule (bean colour vs paper colour, so
   shadows are excluded) decides where the edge is; where two beans touch, a 3-px band is re-drawn by a watershed
   on the photo. Every bean becomes one solid piece, so the mask and the polygon agree.
5. **Keep every bean**: only fragments under 0.3 × the median bean area are dropped (smallest real bean: 0.55 ×).
6. **Class per bean**: a small CNN on a 96 × 96 crop around each bean, touching beans painted out in paper colour,
   white-balanced to the paper colour. All beans go through the network in one batch, on NCNN or OpenCV DNN —
   whichever `warm_up()` measures as faster on the machine.
7. **Before the first photo**: the models run once on `weights/warmup.jpg` and MQTT connects, so photo 1 is not slower.
8. **Robustness**: network calls retry; a photo that cannot be read gets an empty answer instead of stopping the run
   (the training pool contained a broken 1 × 1 PNG); a 409 is never retried; outputs are written after the POST.

## Weights and training
| File | What | Made by |
|---|---|---|
| `weights/best.pt`, `seg.onnx`, `seg_ncnn_model/` | segmenter | `train.py` (two stages, documented in it); `colab/day3_segmenter.ipynb` |
| `weights/bean_cls.onnx`, `bean_cls_ncnn/` | bean classifier | `train_cls_gpu.py final` (`colab/day3_classifier.ipynb`), see below |
| `weights/warmup.jpg` | one Day 3 training photo for the warm-up | — |

Classifier: `gen_cls_day1.py` makes Day 1 bean crops; `train_cls_gpu.py` builds crops from the 16 labelled Day 3
photos and mixed piles of their beans (`gen_cls_day3.py`), trains three networks (seeds 11, 12, 13; 25 epochs,
weights averaged over the last epochs) and exports their average. We use **one** of them (seed 12) — the same
speed as a single network and within 0.3% of the average:
```python
import onnx; m = onnx.load("bean_cls.onnx"); out = [n for n in m.graph.node if n.op_type == "Softmax"][1].output[0]
onnx.utils.extract_model("bean_cls.onnx", "weights/bean_cls.onnx", [m.graph.input[0].name], [out])
# NCNN copy: pnnx weights/bean_cls.onnx inputshape=[1,3,96,96]  ->  weights/bean_cls_ncnn/
```
`python train_cls_gpu.py cv` gives the honest check: each layout's photos held out in turn.

Helpers: `gen_dataset.py` / `gen_dataset_d3.py` (synthetic piles for the segmenter), `day1_train.py` /
`day1_infer.py` (Day 1 crop helpers used by the classifier scripts).
