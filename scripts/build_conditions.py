"""Build four chord conditions from 24 baseline and replay labels."""

import argparse
import json
import random
from pathlib import Path

from chord_conditions import normalize, relation_profile, shifted


def build(record, seed):
    base = [normalize(x) for x in record["baseline"]]
    replay = [normalize(x) for x in record["replay"]]
    if len(base) != 24 or len(replay) != 24:
        raise ValueError("Expected 24 half-second labels per sequence")
    profile, _, _ = relation_profile(base, replay, random.Random(seed))
    central = shifted(base, range(8, 16))
    if base == replay or profile == replay or central == base:
        raise ValueError(
            "Conditions must contain an intervention and a nonidentical profile"
        )
    conditions = dict(
        baseline=base, replay=replay, central_tritone=central, relation_profile=profile
    )
    return [
        dict(
            track_key=record["track_key"],
            condition=name,
            chord_text=" ".join(labels),
            duration=12,
            bpm=120,
            meter=1,
            prompt=record["prompt"],
        )
        for name, labels in conditions.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    jobs = build(json.loads(args.input.read_text()), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jobs, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
