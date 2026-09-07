# MoisesDB observations

Numerical observations only; no source audio, stems, generated audio or weights.
Source dataset: [MoisesDB](https://github.com/moises-ai/moises-db).

| Files | Contents |
|---|---|
| `pilot_*.json` | Six-song pilot, reported separately |
| `validation_*.json` | 24-song evaluation, three seeds per condition |
| `ablation_*.json` | Temporal-only and relation-only observations for those 24 songs |
| `second_recognizer_*.json` | Song-level paired distances using DeepChroma+CRF input recognition |
| `output_chords_*.json` | Seed-level and song-level decoded chord mismatch |
| `accomontage.json` | Song-level 48-beat symbolic-output measurements |

`track_key` identifies the MoisesDB recording. `seed` is 42, 123 or 2027;
seeds are paired, not independent observations. `representation` is CENS or CQT.
`central_tritone` is the central four-second edit; `relation_profile` is joint
matching; `mask_only` is temporal-only; `composition_only` is relation-only.
These field names are retained to preserve compatibility with the saved measurements.

The local train/test folders were a repartition, not an official MoisesDB split.
The evaluation used one recording per artist group, with genre-round-robin selection
and one vocal-active 12-second window per recording, fixed before generation.

`changed_target_effect` measures the difference in relative chord-template
similarity between altered and baseline output. `inside_output_change` and
`full_output_change` measure baseline-to-altered chroma distance. The field
`raw_output_to_replay_distance` measures output chroma-vector distance to replay.
Decoded chord mismatch is a fraction in [0, 1], displayed as a percentage in the README.

Run `python scripts/analyze_moisesdb.py` from the repository root. Distances are
computed after seed averaging; bootstrap resampling and paired tests use songs.
The primary validation models form one Holm family. Second-recognizer comparisons,
target-response ablations and decoded-chord ablations form separate families.
Pilot results are not pooled with validation.

The released data support metric-level reproduction, not re-extraction of features
from audio. All positive and negative songs within each included comparison are retained.
