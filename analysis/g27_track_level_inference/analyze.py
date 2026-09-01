"""Track-level confirmatory inference for the paired G22 comparison."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproduced" / "g27_track_level_inference"
VIEWS = ("stft", "cqt", "cens")
METRICS = (
    # These endpoints are defined for every paired 12-s window and form the
    # frozen six-test confirmatory family.
    ("target_effect", "corrupted_target_effect", "changed_target_effect", "confirmatory"),
    ("inside_output_change", "inside_output_change", "inside_output_change", "confirmatory"),
    # Localization requires at least one unchanged block.  It is retained as
    # a 20-track descriptive diagnostic, never encoded as zero when undefined.
    ("localization_contrast", "localization_contrast", "localization_contrast", "diagnostic"),
)


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def bootstrap_ci(values, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def add_bh(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["wilcoxon_track_level_p"])
    adjusted = [1.0] * len(rows)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(rows) - reverse_rank + 1
        running = min(running, rows[index]["wilcoxon_track_level_p"] * len(rows) / rank)
        adjusted[index] = min(1.0, running)
    for row, value in zip(rows, adjusted):
        row["bh_q"] = value


def main():
    track_rows = []
    summary_rows = []
    for view_index, view in enumerate(VIEWS):
        synthetic_path = (
            ROOT / "data" / "central" / f"{view}.csv"
        )
        real_path = ROOT / "data" / "replay" / f"cnn_crf_{view}.csv"
        synthetic_rows = read(synthetic_path)
        real_rows = read(real_path)
        synthetic = {(row["track_key"], row["seed"]): row for row in synthetic_rows}
        real = {(row["track_key"], row["seed"]): row for row in real_rows}
        if len(synthetic) != len(synthetic_rows) or len(real) != len(real_rows):
            raise RuntimeError(f"{view}: duplicate track-seed key")
        if set(synthetic) != set(real) or len(synthetic) != 90:
            raise RuntimeError(f"{view}: expected 90 exactly matched track-seed rows")
        grouped = defaultdict(lambda: defaultdict(list))
        for key in sorted(synthetic):
            track, seed = key
            for metric_name, synthetic_field, real_field, role in METRICS:
                real_value = real[key][real_field].strip()
                if (
                    metric_name == "localization_contrast"
                    and int(real[key]["changed_blocks"]) == int(real[key]["total_blocks"])
                ):
                    real_value = ""
                if real_value:
                    grouped[track][metric_name].append(
                        float(synthetic[key][synthetic_field]) - float(real_value)
                    )
        if len(grouped) != 30:
            raise RuntimeError(f"{view}: expected 30 tracks")
        for track, metrics in sorted(grouped.items()):
            row = {"representation": view, "track_key": track}
            for metric_name, _, _, _ in METRICS:
                values = metrics[metric_name]
                row[f"synthetic_minus_real_{metric_name}"] = (
                    float(np.mean(values)) if len(values) == 3 else None
                )
            track_rows.append(row)
        for metric_index, (metric_name, _, _, role) in enumerate(METRICS):
            values = [
                row[f"synthetic_minus_real_{metric_name}"]
                for row in track_rows
                if row["representation"] == view
                and row[f"synthetic_minus_real_{metric_name}"] is not None
            ]
            test = wilcoxon(values, alternative="two-sided", method="auto")
            summary_rows.append(
                {
                    "representation": view,
                    "metric": metric_name,
                    "analysis_role": role,
                    "tracks": len(values),
                    "mean_synthetic_minus_real": float(np.mean(values)),
                    "median_synthetic_minus_real": float(np.median(values)),
                    "positive_tracks": int(sum(value > 0 for value in values)),
                    "bootstrap_95_ci_mean": bootstrap_ci(
                        values, seed=20260827 + 100 * view_index + metric_index
                    ),
                    "wilcoxon_track_level_p": float(test.pvalue),
                }
            )
    confirmatory_rows = [row for row in summary_rows if row["analysis_role"] == "confirmatory"]
    add_bh(confirmatory_rows)
    diagnostic_rows = [row for row in summary_rows if row["analysis_role"] == "diagnostic"]
    add_bh(diagnostic_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "track_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=track_rows[0].keys())
        writer.writeheader()
        writer.writerows(track_rows)
    (OUT / "summary.json").write_text(
        json.dumps(
                {
                    "independent_unit": "track",
                    "pairing_unit": "matched track and seed before seed aggregation",
                    "n_tracks": 30,
                    "seeds_per_track": 3,
                    "seed_aggregation": "mean of three paired seed differences within track",
                    "confirmatory_bh_family": "6 tests: 3 representations x changed-target and inside-output endpoints",
                    "diagnostic_bh_family": "3 localization tests over the 20 tracks with both changed and unchanged replay blocks",
                    "localization_definition": "undefined, not zero, when a replay changes all scored blocks",
                    "tests": summary_rows,
                },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
