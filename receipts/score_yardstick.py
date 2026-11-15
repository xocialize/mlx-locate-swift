"""Score locate-gate runs (and mlx-3d's Open Vision Banana PyTorch boxes) against the SAM 3 reference boxes.

  python3 score.py runs/*.jsonl [--ovb /Volumes/Satechi/charm/ovb/out/boxes_all12.json] [--json out.json]

Reference: mlxengine-3d/PROD/mlx-hairdeck-swift/Fixtures/p4b/exNN_box.json (HairDeck's Preprocessor('best'):
SAM 3 hair / face masks; `box` = the head crop, the square around hair ∪ face with an 8 % margin).
Head is scored two ways:
  head_recipe  the same square built from the model's own hair and face boxes (what HairDeck would get);
  head_direct  the square built from the model's box for "the character's head".
Tolerance (AB-R-0851): the head crop within -10 %..+15 % in side and 8 % of the side in centre shift moved the
hair less than another seed did.
"""
import argparse
import json
import os
import statistics as st

FIX = '/Volumes/Satechi/Development/mlxengine-3d/PROD/mlx-hairdeck-swift/Fixtures/p4b'
NAMES = [f'ex{i:02d}' for i in range(1, 13)]
PARTS = {'hair': "the character's hair", 'face': "the character's face", 'head': "the character's head"}


def iou(a, b):
    if not a or not b:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def square(b, margin=0.08):
    if b is None:
        return None
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(b[2] - b[0], b[3] - b[1]) * (1 + 2 * margin)
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def in_tolerance(crop, ref):
    if crop is None:
        return False
    s, sr = crop[2] - crop[0], ref[2] - ref[0]
    ratio = s / sr
    dx = abs((crop[0] + crop[2]) / 2 - (ref[0] + ref[2]) / 2) / sr
    dy = abs((crop[1] + crop[3]) / 2 - (ref[1] + ref[3]) / 2) / sr
    return 0.90 <= ratio <= 1.15 and dx <= 0.08 and dy <= 0.08


def refs():
    return {n: json.load(open(f'{FIX}/{n}_box.json')) for n in NAMES}


def load_gate(path):
    """{name: {part: box_px or None}}, plus timing rows (rep 1 = warm)."""
    boxes, secs, meta = {}, [], {}
    for line in open(path):
        r = json.loads(line)
        if r['event'] == 'load':
            meta['load'] = r
        elif r['event'] == 'done':
            meta['done'] = r
        elif r['event'] == 'box':
            part = next(k for k, v in PARTS.items() if v == r['phrase'])
            px = r['boxes_px'] if 'boxes_px' in r else [b['px'] for b in r['boxes']]
            if r['rep'] == 0:
                boxes.setdefault(r['image'], {})[part] = px[0] if px else None
                boxes[r['image']].setdefault('_n', {})[part] = len(px)
                if 'boxes' in r and r['boxes']:
                    boxes[r['image']].setdefault('_score', {})[part] = r['boxes'][0]['score']
            else:
                secs.append(r['sec'])
    return boxes, secs, meta


def load_ovb(path):
    boxes = {}
    for r in json.load(open(path)):
        boxes[r['name']] = {'hair': r['hair']['box'], 'face': r['face']['box'], 'head': None,
                            '_sec': [r['hair']['sec'], r['face']['sec']]}
    return boxes


def score(boxes, R):
    out = {}
    for part in ('hair', 'face'):
        # A picture whose reference has no such part (ex09: SAM 3 found no face) is left out of the
        # IoU list; whether the model also returned nothing is counted separately.
        keep = [n for n in NAMES if R[n][f'{part}_box'] is not None]
        out[part] = [iou(boxes.get(n, {}).get(part), R[n][f'{part}_box']) for n in keep]
        out[part + '_absent_agree'] = [boxes.get(n, {}).get(part) is None
                                       for n in NAMES if R[n][f'{part}_box'] is None]
    rec = [square(union(boxes.get(n, {}).get('hair'), boxes.get(n, {}).get('face'))) for n in NAMES]
    out['head_recipe'] = [iou(c, R[n]['box']) for c, n in zip(rec, NAMES)]
    out['head_recipe_tol'] = [in_tolerance(c, R[n]['box']) for c, n in zip(rec, NAMES)]
    if any(boxes.get(n, {}).get('head') for n in NAMES):
        d = [square(boxes.get(n, {}).get('head')) for n in NAMES]
        out['head_direct'] = [iou(c, R[n]['box']) for c, n in zip(d, NAMES)]
        out['head_direct_tol'] = [in_tolerance(c, R[n]['box']) for c, n in zip(d, NAMES)]
    return out


def summary(v):
    return f'median {st.median(v):.3f} mean {st.mean(v):.3f} min {min(v):.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--ovb', default='/Volumes/Satechi/charm/ovb/out/boxes_all12.json')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    R = refs()
    table = {}
    if a.ovb and os.path.exists(a.ovb):
        b = load_ovb(a.ovb)
        table['ovb-4b (PyTorch, AB-R-0862)'] = {'scores': score(b, R),
                                                'sec': [s for n in NAMES for s in b[n]['_sec']]}
    for p in a.runs:
        b, secs, meta = load_gate(p)
        table[os.path.basename(p)[:-6]] = {'scores': score(b, R), 'sec': secs, 'meta': meta,
                                           'counts': {n: b.get(n, {}).get('_n') for n in NAMES}}
    for name, t in table.items():
        s = t['scores']
        print(f'## {name}')
        for k in ('hair', 'face', 'head_recipe', 'head_direct'):
            if k in s and s[k]:
                tol = s.get(k + '_tol')
                extra = f' | in tolerance {sum(tol)}/12' if tol else ''
                print(f'  {k:12s} {summary(s[k])}{extra}')
        print('  per image hair', ' '.join(f'{x:.2f}' for x in s['hair']))
        print('  per image face', ' '.join(f'{x:.2f}' for x in s['face']), '(ex09 left out: no reference face)')
        if s['face_absent_agree']:
            print(f"  no-face picture: model also returned none {sum(s['face_absent_agree'])}/{len(s['face_absent_agree'])}")
        if t['sec']:
            print(f"  sec per query: median {st.median(t['sec']):.2f} max {max(t['sec']):.2f}")
        m = t.get('meta', {})
        if m:
            print(f"  load {m['load']['sec']:.1f}s resident(MLX active) {m['load']['mlx_active_gb']} GB, "
                  f"MLX peak {m['done']['mlx_peak_gb']} GB, phys at end {m['done']['phys_gb']} GB")
    if a.json:
        json.dump(table, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
