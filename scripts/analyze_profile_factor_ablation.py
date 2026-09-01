#!/usr/bin/env python3
"""Recompute the full-30 profile-factor replay-distance analysis.

The analysis aggregates diffusion seeds within a condition/profile, then
profiles within a track. It reports paired construction contrasts at the track
level. The construction cells are not a fully orthogonal causal factorial
design, so this program never labels a contrast a causal main effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


CONDITIONS = ("central", "mask_only", "composition_only", "full_profile")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value_from_row(row: dict[str, str], field: str) -> float:
    """Read a metric, deriving full-window output change when only its parts exist."""
    if field in row and row[field] != "":
        return float(row[field])
    if field == "full_output_change":
        inside = float(row["inside_output_change"])
        outside = float(row["outside_output_change"])
        inside_blocks = int(row["inside_blocks"])
        outside_blocks = int(row["outside_blocks"])
        if inside_blocks + outside_blocks == 0:
            raise ValueError("cannot derive full_output_change from an empty window")
        return (inside * inside_blocks + outside * outside_blocks) / (inside_blocks + outside_blocks)
    raise KeyError(f"missing metric {field!r}")


def aggregate(rows: list[dict[str, str]], field: str, condition_prefix: str | None = None,
              central_metric: str | None = None) -> dict[str, float]:
    """Average seeds within each profile then profiles within each track."""
    per_profile: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = row.get("condition_id", "")
        if condition_prefix and not condition.startswith(condition_prefix):
            continue
        profile = condition if condition_prefix else "single"
        value_field = central_metric or field
        per_profile[(row["track_key"], profile)].append(value_from_row(row, value_field))
    per_track: dict[str, list[float]] = defaultdict(list)
    for (track, _), values in per_profile.items():
        per_track[track].append(mean(values))
    return {track: mean(values) for track, values in per_track.items()}


def bootstrap_ci(values: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(draws))
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]


def exact_wilcoxon_two_sided(values: list[float]) -> float:
    """Exact signed-rank p-value with deterministic average ranks for ties."""
    nonzero = [value for value in values if value != 0.0]
    ranked = sorted((abs(value), index) for index, value in enumerate(nonzero))
    doubled_ranks = [0] * len(nonzero)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and math.isclose(ranked[end][0], ranked[start][0], abs_tol=1e-14):
            end += 1
        rank_sum = start + 1 + end
        for _, index in ranked[start:end]:
            doubled_ranks[index] = rank_sum
        start = end
    observed = sum(rank for rank, value in zip(doubled_ranks, nonzero) if value > 0.0)
    total = sum(doubled_ranks)
    lower = min(observed, total - observed)
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in doubled_ranks:
        for subtotal in range(total, rank - 1, -1):
            counts[subtotal] += counts[subtotal - rank]
    return min(1.0, 2.0 * sum(counts[: lower + 1]) / (2 ** len(nonzero)))


def contrast(left: list[float], right: list[float], draws: int, seed: int) -> dict[str, object]:
    gains = [a - b for a, b in zip(left, right)]
    return {
        "mean": mean(gains),
        "median": median(gains),
        "positive_tracks": sum(value > 0.0 for value in gains),
        "ci95": bootstrap_ci(gains, draws, seed),
        "wilcoxon_p": exact_wilcoxon_two_sided(gains),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-profile", type=Path, required=True)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--mask-only", type=Path, required=True)
    parser.add_argument("--composition-only", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--metric", default="changed_target_effect")
    parser.add_argument("--central-metric", default="corrupted_target_effect",
                        help="Metric field for the central-probe CSV. Defaults to its target-effect field.")
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    full = aggregate(read_rows(args.full_profile), args.metric, "profile_matched")
    central = aggregate(read_rows(args.central), args.metric, central_metric=args.central_metric)
    replay = aggregate(read_rows(args.replay), args.metric)
    mask = aggregate(read_rows(args.mask_only), args.metric, "mask_only")
    composition = aggregate(read_rows(args.composition_only), args.metric, "composition_only")
    tracks = sorted(set(full) & set(central) & set(replay) & set(mask) & set(composition))
    if len(tracks) != 30:
        raise ValueError(f"expected 30 paired tracks, found {len(tracks)}")

    distance = {
        "central": [abs(central[track] - replay[track]) for track in tracks],
        "mask_only": [abs(mask[track] - replay[track]) for track in tracks],
        "composition_only": [abs(composition[track] - replay[track]) for track in tracks],
        "full_profile": [abs(full[track] - replay[track]) for track in tracks],
    }
    table = [
        {"track_key": track, **{name: distance[name][index] for name in CONDITIONS}}
        for index, track in enumerate(tracks)
    ]
    contrasts = {
        "composition_gain_at_central_support": contrast(distance["central"], distance["composition_only"], args.draws, args.seed),
        "composition_gain_at_replay_support": contrast(distance["mask_only"], distance["full_profile"], args.draws, args.seed),
        "support_gain_at_tritone_composition": contrast(distance["central"], distance["mask_only"], args.draws, args.seed),
        "support_gain_at_replay_composition": contrast(distance["composition_only"], distance["full_profile"], args.draws, args.seed),
    }
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    qa_pass = qa.get("status_counts") == {"PASS": 360}
    payload = {
        "representation": args.representation,
        "metric": args.metric,
        "tracks": len(tracks),
        "definition": "D(X,R)=abs(E_X-E_replay); seeds average within condition/profile and profiles average within track",
        "central_metric": args.central_metric,
        "mean_distance": {name: mean(distance[name]) for name in CONDITIONS},
        "median_distance": {name: median(distance[name]) for name in CONDITIONS},
        "ci_mean_distance": {name: bootstrap_ci(distance[name], args.draws, args.seed) for name in CONDITIONS},
        "contrasts": contrasts,
        "qa": {"new_renders": qa.get("rows"), "status_counts": qa.get("status_counts"), "all_360_new_renders_pass": qa_pass},
        "interpretation_boundary": "paired construction contrasts, not a factor-wise causal decomposition",
        "track_results": table,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("representation", "tracks", "mean_distance", "qa")}, indent=2))


if __name__ == "__main__":
    main()
