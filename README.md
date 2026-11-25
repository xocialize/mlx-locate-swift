# mlx-locate-swift

A picture and a phrase in, boxes out, on Apple silicon: the model behind MES `image.locate` (AB-A-0293).
It runs [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache-2.0) through
mlx-swift-lm's `MLXVLM`, using its native grounding (boxes on a 0–1000 grid, one entry per instance).

| Product | What |
|---|---|
| `Locate` | Engine-agnostic core: `Locator.load(directory:)`, `locate(image:phrase:maxBoxes:)` → `[LocatedBox]` (`x, y, width, height` as fractions of the upright picture, `label`, `score`), and `answer(image:question:)` |
| `MLXLocate` | The MLXEngine package (`LocatePackage`, `LocateConfiguration`): an `imageLocate` surface (typed boxes) and an `imageAnalysis` surface (mode `answer`, and the older JSON `locate` mode) |
| `locate-cli` | Gates: `yardstick`, `mem-bench`, `engine`, `typed`, `query` |

## Using it through the engine

**`imageLocate`** (engine contract 1.66.0, mlx-engine-swift ≥ 0.81.0; AB-A-0295):

```swift
let response = try await engine.run(
    ImageLocateRequest(image: image, prompt: "the character's hands", count: 2), package: id)
let boxes = (response as! ImageLocateResponse).boxes   // [ImageLocateBox], best first
```

- `count` is the most boxes to return: nil means 1, and above 16 is refused before the run
  (`ImageLocateControls(maxCount: 16, scores: true, seeded: false)`, `LocatePackage.locateControls`).
- The model is always asked for up to 16 boxes, and the answer is `served(count:)`: clipped to the picture, in
  descending score, cut to `count`. So `count: 1` returns the box the model is surest of, not the first one it wrote.
  On a picture with many instances that costs the time of the full list (about 5–8 s for 16 boxes) whatever `count` is.
- Every box has a `label` and a `score`. An empty list is a success. A seed is accepted and ignored.
- `locate-cli typed` runs it through `MLXServeEngine` and exits non-zero unless
  `ImageLocateControls.responseProblems` is empty and a count of 17 is refused (`receipts/typed/`).

**`imageAnalysis`**:

- **mode `answer`**: `prompt` is a question; the response is the model's text.
- **mode `locate`** (or no mode): the carrier from before the typed capability, kept so v0.1.0 and v0.2.0 clients keep
  working. The response text is JSON, `{"boxes": [{"x": 0.27, "y": 0.47, "width": 0.06, "height": 0.07, "label": "the character's hands", "score": 0.63}]}`,
  in the model's own order; `metaData["count"]` (int) caps the list (default 16). New code should use `imageLocate`.

Decoding is greedy, so the same picture and phrase give the same boxes. Transparency is composited over white, and the
longest side is scaled to at most 1024 px (the measured envelope; `LocateConfiguration.maxSide` can lower it, not raise it).
`score` is the geometric mean probability of the tokens that spell the box's coordinates. Use it to rank or threshold;
it is not calibrated (yardstick boxes score 0.28–0.66).

Write the phrase the way you want the box drawn. "The character's face" gives a box from the hairline to the chin;
"the character's facial skin, below the bangs" gives the skin only, which is what SAM 3's face mask covers (median IoU
against it 0.70 → 0.75).

## Lanes and footprint

| quant | weights (pinned) | resident | process peak at the 1024 cap | warm query, 1024 px |
|---|---|---|---|---|
| int8 (default) | `mlx-community/Qwen3-VL-4B-Instruct-8bit` @ `0943db6` | 5.24 GB | 7.59 GB | 0.63 s median |
| int4 | `mlx-community/Qwen3-VL-4B-Instruct-4bit` @ `2fd8dac` | 3.20 GB | 5.61 GB | 0.47 s median |

M5 Max 128 GB, macOS 27.3, Release, mlx-swift 0.32.4 (fleet fork), mlx-swift-lm 3.32.3, 2026-11-15. Re-run on mlx-swift-lm 3.33.0 + mlx-swift 0.33.1 (fleet fork, AB-D-0264) on 2026-11-19: int8 yardstick hair/face/head-crop medians 0.911 / 0.739 / 0.929, head crop 12/12 in tolerance, 0.62 s median query, MLX peak 6.35 GB (`receipts/yardstick-int8-mlxlm333.jsonl`). Resident is
`phys_footprint` after load with the weights evaluated; peak is the ledger peak over one query on a 1022×1024 picture
(1,081 prompt tokens), one process per number, MLX cache uncapped. Query times are the median over 36 queries (12
pictures × hair, face, head). The GPU was shared with other sessions' jobs (60–98 % utilisation before the runs), so a
quiet GPU should be faster.

Those numbers are for one-box answers. A 16-box answer generates about 700 tokens and peaks higher: 8.05 GB (int8) and
6.12 GB (int4) process peak on a 1024×1024 picture of 25 discs, through `MLXServeEngine`, two processes per quant
(2026-11-25, `receipts/typed/`). The manifest declares 5.4 / 3.35 GB resident and 3.0 GB activation to cover that
(2.6 GB before the typed surface). The generation budget grows with the box count and the phrase's length, up to 1,024
tokens; v0.2.0's fixed 256 cut a list after about six boxes.

Weights are evaluated inside `load()`, so the first request reads nothing from disk (the lesson of AB-A-0266).

## Why this model (AB-A-0293)

Yardstick: mlx-3d's 12 HairDeck demo pictures, scored against the SAM 3 boxes of HairDeck's reference preprocessing
(`mlxengine-3d/PROD/mlx-hairdeck-swift/Fixtures/p4b`). "Head crop" builds HairDeck's crop (the square around
hair ∪ face, 8 % margin) from the model's own hair and face boxes. "In tolerance" means within −10 %..+15 % in side and
8 % in centre shift, the range AB-R-0851 found moves the hair less than another seed does.

| model | hair IoU median | face IoU median | head crop IoU median | head crop in tolerance | time per query | memory |
|---|---|---|---|---|---|---|
| **Qwen3-VL-4B int8** (this package) | 0.917 | 0.750 | 0.939 | **12/12** | 0.63 s | 5.2 GB resident |
| Qwen3-VL-4B int4 | 0.913 | 0.721 | 0.940 | 11/12 | 0.47 s | 3.2 GB resident |
| Open Vision Banana 4B (PyTorch boxes, AB-R-0862) | 0.929 | 0.899 | 0.912 | 10/12 | 6.1 s in MLX (4 steps, 512², AB-R-0864) | 16.7 GB MLX peak |
| Qwen3.5-4B 4-bit (mlx-community) | 0.896 | 0.552 | 0.930 | 9/12 | 0.51 s | 3.0 GB |

Open Vision Banana draws a tighter face (its face prompt matches SAM 3's skin-only mask). It also returns a face where
there is none: ex09 shows the back of the head, and Qwen3-VL correctly returns an empty list. It is about 10× slower and
3–5× larger, it gives one mask per prompt where Qwen3-VL gives one box per instance, and its weights are FLUX
Non-Commercial. Receipts: AB-R-0862 and AB-R-0864 (Open Vision Banana); this package's numbers are in `receipts/`
(raw JSONL, `score_yardstick.py`) and on the bridge.

## Licence

Port code: Apache-2.0 (`LICENSE`). Weights: Apache-2.0 (Qwen3-VL-4B-Instruct; the mlx-community conversions carry
the same licence). The training data is undisclosed, so the declared licence governs (AB-D-0105). See `NOTICE`.

## Not covered

- Small or crowded objects, non-character pictures beyond a few spot checks, and pictures larger than 1024 px on a side
  (they are scaled down first).
- More than 16 boxes a request (MES allows 64): not measured, so not declared.
- Stopping the generation early when `count` is below what the picture holds.
- Phrases in languages other than English.
- Qwen3-VL-8B: the upstream bf16 folder on this Mac produced garbage tokens through MLXVLM 3.32.3. Not investigated.
- Timings on a quiet GPU and on a 16 GB Mac.
