#!/usr/bin/env python3
"""Plot harmonic-relation severity and the full-30 construction analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#0F4D92"
RED = "#B64342"
TEAL = "#42949E"
GREY = "#AEB9C3"
INK = "#20262E"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def means_by_track(path: Path, condition: str, metric: str):
    grouped = defaultdict(list)
    for row in rows(path):
        if row["error_type"] == condition:
            grouped[row["track_key"]].append(float(row[metric]))
    return {track: float(np.mean(values)) for track, values in grouped.items()}


def factor_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(ax, label):
    ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-cens", type=Path, required=True)
    parser.add_argument("--factor-cens", type=Path, required=True)
    parser.add_argument("--factor-cqt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    quality = means_by_track(args.relation_cens, "parallel_quality", "full_window_output_error_mass")
    relative = means_by_track(args.relation_cens, "relative_substitution", "full_window_output_error_mass")
    common = sorted(set(quality) & set(relative), key=lambda track: relative[track] - quality[track])
    factor_cens = factor_summary(args.factor_cens)
    factor_cqt = factor_summary(args.factor_cqt)

    # A vertical evidence stack preserves single-column placement while giving
    # the full-30 construction panel enough width for its confidence intervals.
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.10),
                             gridspec_kw={"height_ratios": [0.95, 1.05]})

    ax = axes[0]
    for track in common:
        ax.plot([0, 1], [quality[track], relative[track]], color=GREY, lw=0.65, alpha=0.8)
    ax.scatter(np.zeros(len(common)), [quality[t] for t in common], s=10, color=BLUE,
               edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter(np.ones(len(common)), [relative[t] for t in common], s=10, color=RED,
               edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter([0, 1], [np.mean(list(quality.values())), np.mean(list(relative.values()))],
               s=38, color=[BLUE, RED], edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xticks([0, 1], ["Quality\nflip", "Relative-root\nchange"])
    ax.set_ylabel("CENS full-window output change", labelpad=3)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 0.225)
    ax.set_title("Relative-root changes amplify propagation", loc="left", fontweight="bold", fontsize=10.0, pad=3)
    ax.text(0.98, 0.03, "18/18 tracks higher\n2.88$\\times$ mean response",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.0, color=INK)
    panel_label(ax, "a")

    ax = axes[1]
    conditions = [
        ("central", "Central\ntritone", RED),
        ("mask_only", "Exact\nmask", "#2E9E44"),
        ("composition_only", "Composition\nonly", "#9A4D8E"),
        ("full_profile", "Full\nprofile", TEAL),
    ]
    positions = np.arange(len(conditions))[::-1]
    for offset, (view, summary, marker, edge) in enumerate((
        ("CENS", factor_cens, "o", BLUE),
        ("CQT", factor_cqt, "s", TEAL),
    )):
        for y, (key, _, color) in zip(positions, conditions):
            mean = summary["mean_distance"][key]
            low, high = summary["ci_mean_distance"][key]
            yy = y + (0.13 if offset == 0 else -0.13)
            ax.plot([low, high], [yy, yy], color=color, lw=1.35, solid_capstyle="round", zorder=2)
            ax.scatter(mean, yy, s=25, marker=marker, color=color, edgecolor="white",
                       linewidth=0.45, zorder=3)
    ax.set_yticks(positions, [label for _, label, _ in conditions], fontsize=9.0)
    ax.set_xlim(0, 0.61)
    ax.set_xlabel("Distance to CNN--CRF replay")
    ax.set_title("Support and composition close the replay gap", loc="left", fontweight="bold", fontsize=10.0, pad=3)
    ax.grid(axis="x", color="#E2E6E9", lw=0.55, zorder=0)
    legend_text = r"CENS  $\bullet$    CQT  $\blacksquare$" + "\n" + r"mean $\pm$ 95% CI"
    ax.text(0.03, 0.98, legend_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12})
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.21, right=0.99, top=0.94, bottom=0.14, hspace=0.67)
    save(fig, args.out)


if __name__ == "__main__":
    main()
