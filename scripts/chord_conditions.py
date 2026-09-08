#!/usr/bin/env python3
"""Chord normalization and support-and-relation matching on a half-second grid."""
from __future__ import annotations
import argparse, csv, json, random, hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def segs(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            a, b, l = line.split(maxsplit=2)
            out.append((float(a), float(b), l.split()[0]))
    return out


def at(items, t):
    for a, b, l in items:
        if a <= t < b:
            return l
    raise ValueError(f"no source chord covers sample time {t}")


def clip(items, start, end):
    return [
        (max(a, start) - start, min(b, end) - start, l)
        for a, b, l in items
        if min(b, end) > max(a, start)
    ]


def normalize(label):
    if label in {"N", "X", "", "no_chord", "-", "ε"}:
        return "N"
    root = label.split(":", 1)[0].split("/", 1)[0]
    if root not in PITCH:
        raise ValueError(f"unsupported root: {label}")
    q = label.split(":", 1)[1].split("/", 1)[0].lower() if ":" in label else "maj"
    if q.startswith("min"):
        quality = ":min"
    elif q in {"maj", "major", ""}:
        quality = ""
    else:
        quality = ""  # documented adapter collapses extended quality to triad
    return NAMES[PITCH[root]] + quality


def tokens(items, seconds=12.0):
    return [normalize(at(items, (i + 0.5) * 0.5)) for i in range(round(seconds / 0.5))]


def shifted(base, indices, semitones=6):
    out = []
    for i, l in enumerate(base):
        if i not in indices:
            out.append(l)
            continue
        if l == "N":
            out.append("N")
            continue
        root = l.split(":", 1)[0]
        q = ":min" if ":min" in l else ""
        out.append(NAMES[(PITCH[root] + semitones) % 12] + q)
    return out


def parse(label):
    """Return the MusicGen-Chord root index and major/minor quality."""
    if label == "N":
        return None, "none"
    root = label.split(":", 1)[0]
    return PITCH[root], "minor" if ":min" in label else "major"


def classify(source, target):
    """Use the same relation taxonomy as the MIDI-SAG profile experiment."""
    if source == target:
        return "exact"
    source_root, source_quality = parse(source)
    target_root, target_quality = parse(target)
    if source_root is None or target_root is None:
        return "no_chord_involved"
    if source_root == target_root and source_quality == target_quality:
        return "exact"
    if source_root == target_root:
        return "same_root_quality"
    if (
        source_quality == "major"
        and target_root == (source_root - 3) % 12
        and target_quality == "minor"
    ):
        return "relative_substitution"
    if (
        source_quality == "minor"
        and target_root == (source_root + 3) % 12
        and target_quality == "major"
    ):
        return "relative_substitution"
    interval = (target_root - source_root) % 12
    if interval in {1, 11}:
        return "semitone_root"
    if interval == 6:
        return "tritone_root"
    if interval in {5, 7}:
        return "fifth_root"
    return "other_root"


def label(root, quality):
    return NAMES[root % 12] + (":min" if quality == "minor" else "")


def candidates(source, category):
    """Return every target compatible with a relation category on this grid."""
    root, quality = parse(source)
    if category == "no_chord_involved":
        return ["C", "C:min"] if root is None else ["N"]
    if root is None:
        return []
    if category == "same_root_quality":
        return [label(root, "minor" if quality == "major" else "major")]
    if category == "relative_substitution":
        return (
            [label(root - 3, "minor")]
            if quality == "major"
            else [label(root + 3, "major")]
        )
    shifts = {
        "semitone_root": (1, 11),
        "tritone_root": (6,),
        "fifth_root": (5, 7),
        "other_root": (2, 4, 8, 10),
    }.get(category, ())
    return [
        label(root + shift, target_quality)
        for shift in shifts
        for target_quality in ("major", "minor")
    ]


def relation_profile(base, replay, rng):
    """Match each replay relation; unique targets can coincide with replay."""
    output = []
    categories = []
    copied = 0
    for source, actual in zip(base, replay):
        category = classify(source, actual)
        categories.append(category)
        if category == "exact":
            output.append(source)
            continue
        options = candidates(source, category)
        if not options:
            raise RuntimeError(f"no {category} target available for {source}")
        alternatives = [item for item in options if item != actual]
        target = rng.choice(alternatives or options)
        copied += int(target == actual)
        output.append(target)
    return output, categories, copied
