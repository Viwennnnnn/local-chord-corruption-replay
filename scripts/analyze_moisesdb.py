"""Recompute paired calibration and ablation statistics from released observations."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/moisesdb'
MODELS = ['midi_sag', 'musicgen_chord']

def read(name):
    return json.loads((DATA / name).read_text())

def describe(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0]
    ranks = np.rint(2 * rankdata(abs(nonzero))).astype(int)
    total = int(ranks.sum())
    observed = int(ranks[nonzero > 0].sum())
    threshold = min(observed, total - observed)
    counts = [0] * (total + 1)
    counts[0] = 1
    reachable = 0
    for rank in ranks:
        for j in range(reachable, -1, -1):
            counts[j + rank] += counts[j]
        reachable += rank
    p = sum(c for j, c in enumerate(counts) if min(j, total-j) <= threshold) / 2**len(nonzero)
    samples = np.random.default_rng(20260906).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                positive=int((values > 0).sum()), ci95=np.quantile(samples, [.025, .975]).tolist(), p=p)

def holm(results):
    order = sorted(results, key=lambda k: results[k]['p'])
    previous = 0
    for i, key in enumerate(order):
        previous = max(previous, min(1., results[key]['p'] * (len(order)-i)))
        results[key]['p_holm'] = previous

def distances(rows, representation='cens', endpoint='changed_target_effect'):
    groups = defaultdict(list)
    for r in rows:
        if r['representation'] == representation:
            groups[r['track_key'], r['condition']].append(r)
    means = {}
    for key, group in groups.items():
        assert len(group) == 3 and {r['seed'] for r in group} == {42, 123, 2027}
        means[key] = float(np.mean([r[endpoint] for r in group]))
    tracks = sorted({k[0] for k in means})
    conditions = sorted({k[1] for k in means} - {'replay'})
    return {c: np.array([abs(means[t,c] - means[t,'replay']) for t in tracks]) for c in conditions}

def main():
    report = {}
    for cohort in ['pilot', 'validation']:
        report[cohort] = {}
        for model in MODELS:
            d = distances(read(f'{cohort}_{model}.json')['rows'])
            assert len(d['central_tritone']) == (6 if cohort == 'pilot' else 24)
            report[cohort][model] = dict(central=float(d['central_tritone'].mean()),
                profile=float(d['relation_profile'].mean()), **describe(d['central_tritone']-d['relation_profile']))
        holm(report[cohort])
    report['second_recognizer'] = {}
    report['ablation_response'] = {}
    report['ablation_chords'] = {}
    for model in MODELS:
        rows = [r['changed_target_effect'] for r in read(f'second_recognizer_{model}.json')['track_level'] if r['representation']=='cens']
        report['second_recognizer'][model] = dict(central=float(np.mean([r['central_distance'] for r in rows])),
            profile=float(np.mean([r['profile_distance'] for r in rows])), **describe([r['gain'] for r in rows]))
        d = distances(read(f'validation_{model}.json')['rows'] + read(f'ablation_{model}.json')['rows'])
        for c in ['mask_only','composition_only']:
            report['ablation_response'][f'{model}/{c}'] = dict(single=float(d[c].mean()),
                joint=float(d['relation_profile'].mean()), **describe(d[c]-d['relation_profile']))
        rows = sorted(read(f'output_chords_{model}.json')['track_rows'], key=lambda r:r['track_key'])
        for c in ['mask_only','composition_only']:
            report['ablation_chords'][f'{model}/{c}'] = dict(single=float(np.mean([r[c] for r in rows])),
                joint=float(np.mean([r['relation_profile'] for r in rows])), **describe([r[c]-r['relation_profile'] for r in rows]))
    for family in ['second_recognizer','ablation_response','ablation_chords']:
        holm(report[family])
    native = read('accomontage.json')['tracks']
    report['accomontage'] = dict(central=float(np.mean([r['central_distance'] for r in native])),
        profile=float(np.mean([r['profile_distance'] for r in native])), **describe([r['gain'] for r in native]))
    out = ROOT / 'reproduced'
    out.mkdir(exist_ok=True)
    (out/'moisesdb.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['validation'], indent=2))

if __name__ == '__main__':
    main()
