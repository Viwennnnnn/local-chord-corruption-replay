#!/usr/bin/env python3
"""Compute CENS and CQT responses from generated MusicGen-Chord tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch


def block_chroma(path: Path, sample_rate: int, representation: str) -> np.ndarray:
    wav = torch.load(path, map_location="cpu").float().numpy()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    if representation == "cqt":
        chroma = librosa.feature.chroma_cqt(
            y=wav, sr=sample_rate, hop_length=512, n_octaves=7, bins_per_octave=36
        )
    else:
        chroma = librosa.feature.chroma_cens(
            y=wav, sr=sample_rate, hop_length=512, n_octaves=7, bins_per_octave=36
        )
    frames_per_block = 0.5 * sample_rate / 512.0
    blocks = []
    for index in range(24):
        lo = int(round(index * frames_per_block))
        hi = int(round((index + 1) * frames_per_block))
        vector = chroma[:, lo:hi].mean(axis=1)
        norm = np.linalg.norm(vector)
        blocks.append(vector / norm if norm else np.zeros_like(vector))
    return np.stack(blocks)


ROOTS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def template(label: str) -> np.ndarray:
    if label == "N":
        return np.zeros(12)
    root, quality = (label.split(":", 1) + ["maj"])[:2]
    intervals = (0, 3, 7) if quality.startswith("min") else (0, 4, 7)
    vector = np.zeros(12)
    for interval in intervals:
        vector[(ROOTS[root] + interval) % 12] = 1.0
    return vector / np.linalg.norm(vector)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def condition_endpoints(
    baseline_audio: np.ndarray,
    condition_audio: np.ndarray,
    baseline_labels: list[str],
    condition_labels: list[str],
) -> dict[str, float]:
    changed = [left != right for left, right in zip(baseline_labels, condition_labels)]
    if not any(changed):
        return {
            "changed_target_effect": 0.0,
            "inside_output_change": 0.0,
            "full_output_change": 0.0,
            "changed_blocks": 0,
        }
    target_effects, output_distances = [], []
    for base_audio, cond_audio, base_label, cond_label in zip(
        baseline_audio, condition_audio, baseline_labels, condition_labels
    ):
        source = template(base_label)
        target = template(cond_label)
        baseline_margin = cosine(base_audio, target) - cosine(base_audio, source)
        condition_margin = cosine(cond_audio, target) - cosine(cond_audio, source)
        target_effects.append(condition_margin - baseline_margin)
        output_distances.append(1.0 - cosine(base_audio, cond_audio))
    return {
        "changed_target_effect": float(
            np.mean([v for v, flag in zip(target_effects, changed) if flag])
        ),
        "inside_output_change": float(
            np.mean([v for v, flag in zip(output_distances, changed) if flag])
        ),
        "full_output_change": float(np.mean(output_distances)),
        "changed_blocks": int(sum(changed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--protocol", default="chord_replay")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="analyze the first N manifest tracks; use 1 for a single-song run",
    )
    args = parser.parse_args()

    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = list(dict.fromkeys(job["track_key"] for job in jobs))
    if args.limit is not None:
        tracks = tracks[: args.limit]
    job_lookup = {(job["track_key"], job["condition"]): job for job in jobs}
    baseline_labels = {
        track: job_lookup[(track, "baseline")]["chord_text"].split() for track in tracks
    }
    rows = []
    for track in tracks:
        conditions = sorted(
            {job["condition"] for job in jobs if job["track_key"] == track}
        )
        for seed in args.seeds:
            paths = {}
            for condition in conditions:
                name = f"{track}__{condition}__S{seed}.pt"
                matches = [root / name for root in args.root if (root / name).exists()]
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"expected one generated file for {name}, found {matches}"
                    )
                paths[condition] = matches[0]
            for representation in ("cens", "cqt"):
                features = {
                    condition: block_chroma(path, args.sample_rate, representation)
                    for condition, path in paths.items()
                }
                endpoints = {
                    condition: condition_endpoints(
                        features["baseline"],
                        features[condition],
                        baseline_labels[track],
                        job_lookup[(track, condition)]["chord_text"].split(),
                    )
                    for condition in conditions
                }
                replay = endpoints["replay"]
                for condition in conditions:
                    if condition == "baseline":
                        continue
                    item = {
                        "track_key": track,
                        "seed": seed,
                        "representation": representation,
                        "condition": condition,
                        **endpoints[condition],
                    }
                    if condition != "replay":
                        item.update(
                            {
                                "target_effect_distance_to_replay": abs(
                                    endpoints[condition]["changed_target_effect"]
                                    - replay["changed_target_effect"]
                                ),
                                "inside_output_distance_to_replay": abs(
                                    endpoints[condition]["inside_output_change"]
                                    - replay["inside_output_change"]
                                ),
                                "full_output_distance_to_replay": abs(
                                    endpoints[condition]["full_output_change"]
                                    - replay["full_output_change"]
                                ),
                                "raw_output_to_replay_distance": float(
                                    np.linalg.norm(
                                        features[condition] - features["replay"], axis=1
                                    ).mean()
                                ),
                            }
                        )
                    rows.append(item)

    result = {
        "protocol": args.protocol,
        "metrics": {
            "target_effect_distance_to_replay": "abs(condition-specific changed-target effect minus replay changed-target effect)",
            "inside_output_distance_to_replay": "abs(changed-block output change minus replay changed-block output change)",
            "full_output_distance_to_replay": "abs(full-window output change minus replay full-window output change)",
            "raw_output_to_replay_distance": "secondary mean blockwise chroma-vector distance",
        },
        "independent_unit": "track",
        "seeds": args.seeds,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
