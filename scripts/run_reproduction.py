#!/usr/bin/env python3
"""Regenerate released metric-level statistics and figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run("scripts/audit_g22_pairing.py", "--synthetic", "data/central/cens.csv",
        "--real", "data/replay/cnn_crf_cens.csv", "--output", "reproduced/g22_pairing.json")
    run("analysis/g27_track_level_inference/analyze.py")
    run("scripts/summarize_g15_acr_anisotropy.py", "--metrics", "data/relation_control/stft.csv",
        "--metrics", "data/relation_control/cqt.csv", "--metrics", "data/relation_control/cens.csv",
        "--representations", "stft", "cqt", "cens", "--output-json", "reproduced/g15_relation.json")
    for view in ("cens", "cqt"):
        run("scripts/analyze_profile_matched_replication.py", "--profile", f"data/profile/cnn_full30_{view}.csv",
            "--replay", f"data/replay/cnn_crf_{view}.csv", "--central", f"data/central/{view}.csv",
            "--output", f"reproduced/cnn_{view}.json")
        run("scripts/analyze_profile_matched_replication.py", "--profile", f"data/profile/deepchroma_full29_profile_{view}.csv",
            "--replay", f"data/profile/deepchroma_full29_replay_{view}.csv", "--central", f"data/central/{view}.csv",
            "--output", f"reproduced/deep_{view}.json")
        run("scripts/analyze_profile_factor_ablation.py", "--full-profile", f"data/profile/cnn_full30_{view}.csv",
            "--central", f"data/central/{view}.csv", "--replay", f"data/replay/cnn_crf_{view}.csv",
            "--mask-only", f"data/construction/mask_only_{view}.csv", "--composition-only", f"data/construction/composition_only_{view}.csv",
            "--qa", "data/construction/audio_qa.json", "--representation", view,
            "--output-csv", f"reproduced/construction_{view}.csv", "--output-json", f"reproduced/construction_{view}.json")
    run("scripts/finalize_profile_construction_statistics.py", "--cens", "reproduced/construction_cens.json",
        "--cqt", "reproduced/construction_cqt.json", "--output-json", "reproduced/construction_statistics.json",
        "--output-markdown", "reproduced/CONSTRUCTION_STATISTICS.md")
    run("scripts/plot_fig1_calibrated_protocol.py", "--central", "data/central/cens.csv",
        "--replay", "data/replay/cnn_crf_cens.csv", "--cnn-summary", "reproduced/cnn_cens.json",
        "--cnn-cqt-summary", "reproduced/cnn_cqt.json", "--deep-summary", "reproduced/deep_cens.json",
        "--deep-cqt-summary", "reproduced/deep_cqt.json", "--out", "reproduced/figures/fig1_calibrated_protocol")
    run("scripts/plot_fig2_relation_seed.py", "--relation-cens", "data/relation_control/cens.csv",
        "--factor-cens", "reproduced/construction_cens.json", "--factor-cqt", "reproduced/construction_cqt.json",
        "--out", "reproduced/figures/fig2_relation_seed")
    print("PASS: metric-level reproduction complete")


if __name__ == "__main__":
    main()
