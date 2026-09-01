#!/usr/bin/env python3
"""Build the manuscript hero figure for calibrated ACR-to-SAG evaluation."""

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


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 9.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "legend.frameon": False,
})

PALETTE = {
    "central": "#B64342",
    "replay": "#0F4D92",
    "profile": "#42949E",
    "mask": "#2E9E44",
    "composition": "#9A4D8E",
    "neutral": "#8A8A8A",
    "grid": "#D9D9D9",
    "ink": "#20262E",
    "paper": "#F5F7F8",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def track_means(path: Path, metric: str):
    grouped = defaultdict(list)
    for row in read_csv(path):
        grouped[row["track_key"]].append(float(row[metric]))
    return {track: float(np.mean(values)) for track, values in grouped.items()}


def label_panel(ax, label: str):
    ax.text(-0.09, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9.5,
            ha="left", va="bottom")


def bootstrap_ci(values, seed=2027, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        sample = rng.choice(values, size=(stop - start, len(values)), replace=True)
        means[start:stop] = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summary_stats(payload):
    # CNN summaries keep the same rows under ``track_results``; the independently
    # recomputed DeepChroma files use ``results``.  Support both audited schemas.
    rows = payload.get("track_results", payload.get("results"))
    if rows is None:
        raise KeyError("expected track_results or results in calibration summary")
    values = [float(row["primary_contrast"]) for row in rows]
    mean = float(np.mean(values))
    low, high = bootstrap_ci(values)
    positive = sum(value > 0 for value in values)
    return mean, low, high, positive, len(values)


def draw_protocol(ax):
    """Draw the evaluation logic before presenting the quantitative result."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.90, "Same vocal, track, seed, context, and scoring window",
            fontsize=9.0, color=PALETTE["neutral"], va="center")
    rows = [
        ("Baseline", [0, 0, 0, 0, 0, 0, 0, 0], PALETTE["neutral"]),
        ("Central 4-s tritone", [0, 0, 1, 1, 1, 0, 0, 0], PALETTE["central"]),
        ("Complete recognizer replay", [1, 0, 1, 1, 0, 1, 0, 1], PALETTE["replay"]),
        ("Support+profile surrogate", [1, 0, 1, 1, 0, 1, 0, 1], PALETTE["profile"]),
    ]
    y_positions = [0.70, 0.50, 0.30, 0.10]
    x0, block_w, gap = 0.32, 0.032, 0.006
    for (label, mask, color), y in zip(rows, y_positions):
        ax.text(0.015, y + 0.045, label, fontsize=9.0, fontweight="bold",
                color=color if label != "Baseline" else PALETTE["ink"], va="center")
        for index, changed in enumerate(mask):
            face = color if changed else "#E7EBEE"
            edge = color if changed else "#CDD3D8"
            ax.add_patch(plt.Rectangle((x0 + index * (block_w + gap), y), block_w, 0.09,
                                       facecolor=face, edgecolor=edge, lw=0.55))

    generator_x = 0.70
    ax.add_patch(plt.Rectangle((generator_x, 0.20), 0.15, 0.48,
                               facecolor=PALETTE["paper"], edgecolor="#9EA8B0", lw=0.75))
    ax.text(generator_x + 0.075, 0.48, "fixed\nMIDI-SAG", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
    for y in y_positions:
        ax.annotate("", xy=(generator_x, y + 0.045), xytext=(0.65, y + 0.045),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "#9EA8B0"})

    ax.annotate("", xy=(0.93, 0.44), xytext=(0.86, 0.44),
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": PALETTE["ink"]})
    ax.text(0.99, 0.62, "paired output\nresponse", ha="right", va="center",
            fontsize=9.0, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.99, 0.22, "probe severity\nvs. replay proximity", ha="right", va="center",
            fontsize=9.0, color=PALETTE["neutral"])


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cnn-summary", type=Path, required=True)
    parser.add_argument("--cnn-cqt-summary", type=Path, required=True)
    parser.add_argument("--deep-summary", type=Path, required=True)
    parser.add_argument("--deep-cqt-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    central = track_means(args.central, "corrupted_target_effect")
    replay = track_means(args.replay, "changed_target_effect")
    tracks = sorted(set(central) & set(replay), key=lambda track: central[track] - replay[track])
    differences = np.array([central[track] - replay[track] for track in tracks])

    cnn = json.loads(args.cnn_summary.read_text(encoding="utf-8"))
    cnn_cqt = json.loads(args.cnn_cqt_summary.read_text(encoding="utf-8"))
    deep = json.loads(args.deep_summary.read_text(encoding="utf-8"))
    deep_cqt = json.loads(args.deep_cqt_summary.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.25, 3.18))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.92, 2.02], hspace=0.36)

    # a: protocol schematic establishes the condition and color vocabulary.
    ax = fig.add_subplot(outer[0])
    draw_protocol(ax)
    ax.set_title("Local probe versus complete recognizer replay",
                 loc="left", fontweight="bold", pad=2, fontsize=10.0)
    label_panel(ax, "a")

    grid = outer[1].subgridspec(1, 2, width_ratios=[1.58, 1.32], wspace=0.42)

    # b: primary 30-track paired evidence
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(tracks))
    for yi, track in zip(y, tracks):
        ax.plot([replay[track], central[track]], [yi, yi], color="#B9C6D4", lw=0.7, zorder=1)
    ax.scatter([replay[track] for track in tracks], y, s=16, color=PALETTE["replay"],
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter([central[track] for track in tracks], y, s=16, color=PALETTE["central"],
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color=PALETTE["neutral"], lw=0.7, ls=":")
    ax.set_ylim(-1, len(tracks))
    ax.set_yticks([])
    ax.set_xlabel("CENS changed-target effect")
    ax.set_title("Local probe magnifies propagation", loc="left", fontweight="bold", pad=4, fontsize=10.0)
    # The lower-left margin is clear once y labels are omitted; using it keeps
    # the compact result note away from the large central-probe observations.
    ax.text(0.04, 0.16, "29/30 central > replay\nmean gap = 0.462",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0,
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.22"})
    ax.text(0.04, 0.97, "● CNN--CRF replay", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color=PALETTE["replay"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
    ax.text(0.04, 0.89, "● central 4-s tritone", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color=PALETTE["central"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
    label_panel(ax, "b")

    # c: recognizer-path confirmation, with no implied generator replication.
    ax = fig.add_subplot(grid[1])
    rows = [
        ("CNN--CRF", "CENS", cnn, "o"),
        ("CNN--CRF", "CQT", cnn_cqt, "s"),
        ("DeepChroma+CRF", "CENS", deep, "o"),
        ("DeepChroma+CRF", "CQT", deep_cqt, "s"),
    ]
    y_values = [3.2, 2.4, 1.1, 0.3]
    for yi, (recognizer, view, payload, marker) in zip(y_values, rows):
        mean, low, high, positive, n_tracks = summary_stats(payload)
        color = PALETTE["replay"] if recognizer == "CNN--CRF" else PALETTE["profile"]
        ax.plot([low, high], [yi, yi], color=color, lw=1.7, solid_capstyle="round")
        ax.scatter(mean, yi, s=31, marker=marker, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(-0.02, yi, f"{recognizer}\n{view}", ha="right", va="center",
                transform=ax.get_yaxis_transform(), fontsize=9.0, linespacing=0.95)
        ax.text(high + 0.012, yi, f"{positive}/{n_tracks}", va="center", fontsize=9.0,
                color=PALETTE["neutral"])
    ax.axvline(0, color=PALETTE["neutral"], lw=0.7, ls=":")
    ax.set_ylim(-0.3, 3.8)
    ax.set_yticks([])
    ax.set_xlim(-0.03, 0.58)
    ax.set_xlabel("Calibration gain, $\\Delta_{profile}$")
    ax.set_title("Profile matching narrows replay mismatch", loc="left", fontweight="bold", pad=4, fontsize=10.0)
    label_panel(ax, "c")

    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.15)
    save(fig, args.out)


if __name__ == "__main__":
    main()
