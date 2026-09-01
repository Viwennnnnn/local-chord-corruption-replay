#!/usr/bin/env python3
"""Finalize the declared Holm family for the full-30 construction study.

The construction experiment contains four paired contrasts in each of
two paired feature views (CENS and CQT).  This utility reads their raw exact
Wilcoxon p-values from the source-of-record analyses, applies Holm's step-down
procedure once across all eight tests, and writes a compact, citable record.
It does not recompute any audio metric or reinterpret the cells as causal
factor effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY = {
    "composition_gain_at_central_support": "Composition match at central support",
    "composition_gain_at_replay_support": "Full profile over exact-mask-only",
    "support_gain_at_tritone_composition": "Exact support at tritone composition",
    "support_gain_at_replay_composition": "Full profile over composition-only",
}


def holm(values: list[float]) -> list[float]:
    """Return Holm-adjusted values in the original order."""
    count = len(values)
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cens", type=Path, required=True)
    parser.add_argument("--cqt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    analyses = {"CENS": json.loads(args.cens.read_text(encoding="utf-8")),
                "CQT": json.loads(args.cqt.read_text(encoding="utf-8"))}
    names = list(DISPLAY)
    records = []
    for view in ("CENS", "CQT"):
        for name in names:
            contrast = analyses[view]["contrasts"][name]
            records.append({
                "view": view,
                "contrast_id": name,
                "contrast": DISPLAY[name],
                "raw_wilcoxon_p": contrast["wilcoxon_p"],
                "mean_gain": contrast["mean"],
                "positive_tracks": contrast["positive_tracks"],
                "tracks": analyses[view]["tracks"],
            })
    for record, adjusted in zip(records, holm([row["raw_wilcoxon_p"] for row in records])):
        record["holm_adjusted_p"] = adjusted

    payload = {
        "analysis": "full30_profile_construction_holm_v1",
        "family_definition": "Eight paired construction contrasts: four contrasts in each of the paired CENS and CQT views.",
        "test": "two-sided exact Wilcoxon signed-rank test at the track level; diffusion seeds average within condition/profile before inference",
        "multiplicity_adjustment": "Holm step-down across the eight-test family",
        "interpretation_boundary": "Construction contrasts assess matched-condition fidelity; they are not factor-wise causal effects of the generator.",
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Full-30 construction-study statistical family", "",
        "The manuscript's construction results use one declared family of eight paired tests: four construction contrasts in each paired feature view (CENS and CQT). Each raw value is the exact two-sided Wilcoxon signed-rank test over 30 track means; seeds are averaged before the test. Holm's step-down procedure is applied once across all eight values.",
        "",
        "This correction addresses multiplicity within the diagnostic construction analysis only. The cells are paired constructions, not a causal factorial decomposition of MIDI-SAG.",
        "",
        "| View | Construction contrast | Mean replay-distance gain | Positive tracks | Raw $p$ | Holm $p$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['view']} | {row['contrast']} | {row['mean_gain']:.6f} | "
            f"{row['positive_tracks']}/{row['tracks']} | {row['raw_wilcoxon_p']:.6g} | {row['holm_adjusted_p']:.6g} |"
        )
    lines.append("")
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
