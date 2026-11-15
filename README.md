# mlx-locate-swift

A picture and a phrase in, boxes out, on Apple silicon: the model behind MES `image.locate` (AB-A-0293).
It runs [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache-2.0) through
mlx-swift-lm's `MLXVLM`, using its native grounding (boxes on a 0–1000 grid, one entry per instance).

| Product | What |
|---|---|
| `Locate` | Engine-agnostic core: `Locator.load(directory:)`, `locate(image:phrase:maxBoxes:)` → `[LocatedBox]` (`x, y, width, height` as fractions of the upright picture, `label`, `score`), and `answer(image:question:)` |
| `MLXLocate` | The MLXEngine package (`LocatePackage`, `LocateConfiguration`): an `imageAnalysis` surface with modes `locate` (default) and `answer` |
| `locate-cli` | Gates: `yardstick`, `mem-bench`, `engine`, `query` |

## Using it through the engine

The engine contract has no box-typed capability yet, so the package serves `imageAnalysis`:

- **mode `locate`** (or no mode): `prompt` is the phrase. The response text is JSON in the `image.locate` output shape:
  `{"boxes": [{"x": 0.27, "y": 0.47, "width": 0.06, "height": 0.07, "label": "the character's hands", "score": 0.63}]}`.
  `metaData["count"]` (int) caps the number of boxes (default 16). An empty list is a success.
- **mode `answer`**: `prompt` is a question; the response is the model's text.

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

M5 Max 128 GB, macOS 27.3, Release, mlx-swift 0.32.4 (fleet fork), mlx-swift-lm 3.32.3, 2026-11-15. Resident is
`phys_footprint` after load with the weights evaluated; peak is the ledger peak over one query on a 1022×1024 picture
(1,081 prompt tokens), one process per number, MLX cache uncapped. Query times are the median over 36 queries (12
pictures × hair, face, head). The GPU was shared with other sessions' jobs (60–98 % utilisation before the runs), so a
quiet GPU should be faster. The manifest declares 5.4 / 3.35 GB resident and 2.6 GB activation.

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
- A typed `image.locate` capability in the engine (asked of Gaea and mlx-engine on AB-A-0293).
- Qwen3-VL-8B: the upstream bf16 folder on this Mac produced garbage tokens through MLXVLM 3.32.3. Not investigated.
- Timings on a quiet GPU and on a 16 GB Mac.
