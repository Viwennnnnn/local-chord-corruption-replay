import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chord_conditions import classify, normalize, relation_profile, shifted
from build_conditions import build


class ConditionsTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize("Db:min7"), "C#:min")
        self.assertEqual(normalize("X"), "N")

    def test_no_chord_is_unchanged(self):
        self.assertEqual(classify("N", "N"), "exact")
        self.assertEqual(shifted(["N", "C"], [0, 1]), ["N", "F#"])

    def test_profile_preserves_each_relation(self):
        base = ["C", "N", "C", "C", "A:min"]
        replay = ["G", "N", "C:min", "F#", "C"]
        profile, categories, _ = relation_profile(base, replay, random.Random(42))
        self.assertEqual(categories, [classify(a, b) for a, b in zip(base, profile)])
        self.assertNotEqual(profile, replay)

    def test_four_conditions(self):
        record = dict(
            track_key="example",
            baseline=["C"] * 24,
            replay=["G"] * 24,
            prompt="instrumental piano",
        )
        jobs = build(record, 42)
        self.assertEqual(len(jobs), 4)
        labels = jobs[2]["chord_text"].split()
        self.assertEqual(labels[8:16], ["F#"] * 8)
        self.assertEqual(labels[:8] + labels[16:], ["C"] * 16)


if __name__ == "__main__":
    unittest.main()
