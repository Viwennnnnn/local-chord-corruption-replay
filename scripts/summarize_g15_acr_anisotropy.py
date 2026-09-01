#!/usr/bin/env python3
"""Summarize the paired 18-song G15 error-anisotropy confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


TYPES = ("parallel_quality", "relative_substitution")
METRICS = (
    "full_window_input_error_mass",
    "full_window_output_error_mass",
    "changed_block_target_effect",
    "localization_contrast",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def bootstrap_ci(values: list[float], seed: int = 20260823) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.mean(generator.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--representations", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if len(args.metrics) != len(args.representations):
        raise ValueError("metrics and representations must have equal lengths")

    summary = {"protocol": "g15_acr_anisotropy_confirm_v1", "representations": {}}
    for representation, path in zip(args.representations, args.metrics):
        rows = read_csv(path)
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["track_key"], row["error_type"])].append(row)
        tracks = sorted({track for track, _ in grouped})
        if len(rows) != 108 or len(tracks) != 18:
            raise ValueError(f"{representation}: expected 108 rows/18 tracks, got {len(rows)}/{len(tracks)}")
        track_values = {}
        for track in tracks:
            track_values[track] = {}
            for error_type in TYPES:
                bucket = grouped[(track, error_type)]
                if len(bucket) != 3:
                    raise ValueError(f"{representation}/{track}/{error_type}: expected three seeds")
                track_values[track][error_type] = {
                    metric: float(np.mean([float(row[metric]) for row in bucket]))
                    for metric in METRICS
                }
        differences = {
            metric: [
                track_values[track]["relative_substitution"][metric]
                - track_values[track]["parallel_quality"][metric]
                for track in tracks
            ]
            for metric in METRICS
        }
        type_means = {
            error_type: {
                metric: float(np.mean([track_values[track][error_type][metric] for track in tracks]))
                for metric in METRICS
            }
            for error_type in TYPES
        }
        difference_summary = {
            metric: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "positive_tracks": int(sum(value > 0 for value in values)),
                "bootstrap_95_ci_mean": bootstrap_ci(values),
                "wilcoxon_two_sided_p": float(wilcoxon(values).pvalue) if any(values) else 1.0,
            }
            for metric, values in differences.items()
        }
        input_abs_difference = [abs(value) for value in differences["full_window_input_error_mass"]]
        output_delta = difference_summary["full_window_output_error_mass"]
        gates = {
            "input_balance_mean_abs_le_0_02": float(np.mean(input_abs_difference)) <= 0.02,
            "relative_output_wins_at_least_13_of_18": output_delta["positive_tracks"] >= 13,
            "output_mass_ci_excludes_zero": output_delta["bootstrap_95_ci_mean"][0] > 0,
            "positive_target_difference": difference_summary["changed_block_target_effect"]["mean"] > 0,
        }
        summary["representations"][representation] = {
            "tracks": 18,
            "type_means": type_means,
            "paired_relative_minus_parallel": difference_summary,
            "mean_abs_input_mass_difference": float(np.mean(input_abs_difference)),
            "gates": gates,
        }

    reps = summary["representations"]
    summary["global_gates"] = {
        "complete_three_representations": set(reps) == {"stft", "cqt", "cens"},
        "input_balance_all": all(x["gates"]["input_balance_mean_abs_le_0_02"] for x in reps.values()),
        "output_wins_all": all(x["gates"]["relative_output_wins_at_least_13_of_18"] for x in reps.values()),
        "output_ci_excludes_zero_at_least_two": sum(
            x["gates"]["output_mass_ci_excludes_zero"] for x in reps.values()
        ) >= 2,
        "target_difference_positive_all": all(x["gates"]["positive_target_difference"] for x in reps.values()),
    }
    summary["confirm_anisotropy"] = all(summary["global_gates"].values())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "representations": {
            name: {
                "input_abs_delta": values["mean_abs_input_mass_difference"],
                "parallel_output": values["type_means"]["parallel_quality"]["full_window_output_error_mass"],
                "relative_output": values["type_means"]["relative_substitution"]["full_window_output_error_mass"],
                "output_delta": values["paired_relative_minus_parallel"]["full_window_output_error_mass"],
                "target_delta": values["paired_relative_minus_parallel"]["changed_block_target_effect"],
            }
            for name, values in reps.items()
        },
        "global_gates": summary["global_gates"],
        "confirm_anisotropy": summary["confirm_anisotropy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
