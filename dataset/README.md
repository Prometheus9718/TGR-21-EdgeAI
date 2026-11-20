**Download:** [TGR-21_EdgeAI_dataset.zip](https://github.com/Prometheus9718/TGR-21-EdgeAI/releases/latest/download/TGR-21_EdgeAI_dataset.zip) (49 MB, under Releases). This is the README inside it.

# TGR-21 Edge AI dataset — TESA Top Gun Rally 2026

Data we generated and results we produced in the Edge AI track (Chumphon Robusta coffee beans), by team TGR-21 RTNA (Royal Thai Naval Academy). The code that made it is in `scripts/` and at https://github.com/Prometheus9718/TGR-21-EdgeAI.

## What's inside

| Folder | What it is | Photos | Beans |
|---|---|---|---|
| `simulated_piles/random_piles/` | Tight mixed piles of 10–31 real Day 3 beans, cut from all 16 labelled photos, with exact masks and classes | 12 | 226 |
| `simulated_piles/triangle_piles/` | Piles of 8–32 beans cut only from the five "triangle" photos, which our Day 3 segmenter never trained on | 12 | 204 |
| `stress_tests/` | 6 photos with 100–155 beans (3 grids, 3 dense piles), and 3 with the camera farther back: a dense pile shrunk to 0.6× in the same frame, 30–51 beans visible enough to label | 9 | 871 |
| `test_result/` | Our outlines on the real 68-bean test photo, and every bean grouped by the class we gave it | 3 images | 68 |
| `scripts/` | The generators: these piles, the 900-pile training set, classifier crops | | |

## Formats

- `images/*.jpg` (and the `.jpg` files in `stress_tests/`): 3072 × 4096 photos, the same size as the real ones.
- `gt/*.png` (and the `.png` files in `stress_tests/`): 16-bit id maps. 0 = paper, k = bean k.
- `labels.json`: `{"photo name": ["class of bean 1", "class of bean 2", ...]}`, in bean-id order.
- Classes: `good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`.

## How the piles were made

1. Each bean was cut out of the labelled Day 3 photos using the organisers' hand-drawn outlines, and the paper behind it was filled in.
2. Beans were rotated, scaled slightly (±7%), colour-jittered a little and pasted into tight piles with soft shadows; beans touch and overlap.
3. A bean less than 25% visible was left out of the ground truth.
4. Piles were made at half size and upscaled to 3072 × 4096.

Scripts: `make_d3_piles.py` and `make_d3_piles_tri.py` (these two sets; they expect `gen_dataset.py` and the organisers' photos at the paths inside), `gen_dataset_d3.py` (the 900-pile set the Day 3 segmenter was fine-tuned on; rerun it to regenerate, it is not included), `gen_cls_day3.py` (classifier crops), `eval_share.py` (the scoring below; it expects our code and weights at the paths inside).

## Our results on these sets

Final build (YOLO26n-seg at 640 on NCNN, edge refinement, per-bean classifier), scored with the judges' rules on a PC.

| Set | Photos | Beans | Found | Outlines with IoU > 0.90 | Mean IoU | Correct class |
|---|---|---|---|---|---|---|
| 16 real Day 3 photos (organisers' outlines) | 16 | 124 | 100% | 100% | 0.967 | 97.6% |
| … of which the 5 triangle photos | 5 | 30 | 100% | 100% | 0.965 | 96.7% |
| `random_piles` | 12 | 226 | 96.9% | 93.4% | 0.920 | 90.9% |
| `triangle_piles` | 12 | 204 | 96.1% | 87.7% | 0.914 | 90.8% |
| `stress_tests`, 100–155 beans | 6 | 744 | 97.3% | 86.3% | 0.919 | 91.4% |
| `stress_tests`, camera farther back | 3 | 127 | 90.6% | 57.5% | 0.817 | 68.7% |

- **Found:** the bean is matched one-to-one to one of our outlines with IoU > 0.5.
- **Outlines with IoU > 0.90:** the segmentation rule on Day 3.
- **Mean IoU:** matched IoUs averaged over the larger of (true beans, our outlines), so a missed bean or an extra outline counts as 0.
- **Correct class:** share of found beans given the right class (the Day 3 classification rule).
- The classifier trained on beans from all 16 real photos, and every image here is built from those beans, so the class column is optimistic. On Day 1 beans it never saw, it got about 89%.
- The segmenter trained on the grid and square photos and piles of their beans; the triangle photos and piles of their beans were held out.

On the real test (one 68-bean photo, 20 Nov 2026), the pipeline found all 68 beans at 403 ms per photo on a Raspberry Pi 5 and scored 53.01 / 60, 1st place in Edge AI Integration.

## Credit

Every bean in these images comes from photos provided by the TESA Top Gun Rally 2026 organisers: the piles are composites of their beans, and `test_result/` shows their test photo. They are published with the organisers' permission. The scripts and the numbers are ours.
