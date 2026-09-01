# Local chord corruption and recognizer replay

Small, reproducible Python analyses for comparing localized chord edits with
automatic chord-recognizer sequences in singing accompaniment generation.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python scripts/run_reproduction.py
```

The command reads the included metric tables, recomputes summary statistics,
and writes fresh files to `reproduced/` (ignored by Git). Selected figures are
in [`results/figures/`](results/figures/).

## Layout

- `scripts/` contains the analysis and plotting programs.
- `analysis/` contains the track-level inference routine used by the runner.
- `data/` contains the metric tables used as program inputs.
- `results/` contains the final figures and a compact summary.

Audio, model checkpoints, and third-party recognizer implementations are not
included and remain subject to their original licenses.

## License

Code is released under the MIT License. Derived tables and figures are released
under CC BY 4.0; see [DATA_LICENSE.md](DATA_LICENSE.md).
