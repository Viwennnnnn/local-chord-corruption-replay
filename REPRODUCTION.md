# Reproduction

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
