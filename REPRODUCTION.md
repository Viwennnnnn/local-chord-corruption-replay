# Reproduction

For new audio generation, see the [four-condition example](examples/README.md)
and [model installation](environments/README.md). This page covers the released
statistical analyses.

Install the dependencies and run:

```bash
python scripts/run_reproduction.py
```

The runner produces the statistics and figures used in the repository. Results
are written to `reproduced/`, so the working tree stays clean. The independent
unit for paired analyses is the track; the three generation seeds are averaged
within each track before inference.

Only derived metric tables are distributed. Audio-level regeneration requires
the original dataset, checkpoints, and recognizer software.

## Cross-model evaluation

`python scripts/analyze_moisesdb.py` writes `reproduced/moisesdb.json` with the
six-song pilot, 24-song primary evaluation, second-recognizer comparison,
single-factor ablations, decoded-chord comparisons and AccoMontage evaluation.
See [the data dictionary](data/moisesdb/README.md) for fields and units.

Expected validation CENS results:

| Model | Mean gain | Positive songs | Holm-adjusted p |
|---|---:|---:|---:|
| MIDI-SAG | 0.3693558 | 24/24 | 2.3841858e-7 |
| MusicGen-Chord | 0.3370496 | 24/24 | 2.3841858e-7 |

Bootstrap intervals use 10,000 draws with seed 20260906. Exact signed-rank tests
enumerate the sign-flip distribution by dynamic programming, including tied ranks.
Floating-point differences across library versions should not affect the displayed results.
