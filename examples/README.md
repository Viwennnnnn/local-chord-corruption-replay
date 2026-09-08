# A four-condition example

`chords.json` is an artificial C-major/G-major example, not a dataset sample.
It contains 24 labels per sequence, one label per half second. Supply your own
baseline and recognized replay labels in the same format.

```bash
python scripts/build_conditions.py --input examples/chords.json --out outputs/example_manifest.json
```

This creates baseline, complete replay, central four-second tritone and a matched
profile. The profile uses the same change locations and per-cell relation classes
as replay. Its random seed controls chord choice, not the generation seed.

After installing the MusicGen-Chord environment:

```bash
python scripts/generate_musicgen.py \
  --repo external/musicgen-chord \
  --weights checkpoints/musicgen-stereo-chord.th \
  --t5 checkpoints/t5-base --encodec checkpoints/encodec_32khz \
  --manifest outputs/example_manifest.json --out outputs/example \
  --seeds 42 123 2027 --device cuda --repeat-check

python scripts/score_musicgen.py \
  --root outputs/example --manifest outputs/example_manifest.json \
  --seeds 42 123 2027 --out outputs/example_metrics.json
```

Expect 12 `.pt` waveform tensors and one generation summary, then CENS and CQT
endpoint rows in `example_metrics.json`. Waveforms retain the model's floating-point
values. Each tensor has shape `[1, channels, samples]` at 32 kHz.

The interface uses `bpm=120, meter=1` to place each token at a half-second interval;
this is a control-grid encoding, not an estimate of a song's meter.

For statistical comparison, average `changed_target_effect` across seeds within
each condition first. Then calculate
`abs(central - replay) - abs(profile - replay)` for each song. The per-seed
distance columns are diagnostics, not the primary aggregate statistic.

Tests without a GPU:

```bash
python -m unittest discover -s tests -v
```
