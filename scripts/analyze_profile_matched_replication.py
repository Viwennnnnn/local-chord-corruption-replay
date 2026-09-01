#!/usr/bin/env python3
"""Analyze profile-matched calibration against an alternate recognizer replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def by_track(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        out[row["track_key"]].append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--central-field", default="corrupted_target_effect")
    parser.add_argument("--metric-field", default="changed_target_effect",
                        help="Shared profile/replay metric column; central-field is separate for legacy central files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_seed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read(args.profile):
        profile_seed[(row["track_key"], row["condition_id"])].append(float(row[args.metric_field]))
    profile: dict[str, list[float]] = defaultdict(list)
    for (track, _), values in profile_seed.items():
        profile[track].append(float(np.mean(values)))
    profile_mean = {track: float(np.mean(values)) for track, values in profile.items()}
    replay = by_track(read(args.replay), args.metric_field)
    central = by_track(read(args.central), args.central_field)
    table = []
    for track in sorted(profile_mean):
        contrast = abs(central[track] - replay[track]) - abs(profile_mean[track] - replay[track])
        table.append({"track_key": track, "profile_effect": profile_mean[track], "replay_effect": replay[track],
                      "central_effect": central[track], "primary_contrast": contrast})
    values = np.array([row["primary_contrast"] for row in table])
    rng = np.random.default_rng(20260826)
    boots = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    result = {"tracks": len(table), "track_results": table, "mean_primary_contrast": float(np.mean(values)),
              "bootstrap_95_ci": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
              "positive_tracks": int(np.sum(values > 0)), "wilcoxon_two_sided_p": float(wilcoxon(values).pvalue)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
