from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imed_nvs_submission.predict import _sequence_dirs, _stage_source_only_sequence, new_view
from imed_nvs_submission.validate_output import validate_output_tree


def _make_sequence(root: Path, name: str = "sequence_a", frames: int = 2) -> Path:
    sequence = root / name
    (sequence / "endoscope2" / "L").mkdir(parents=True)
    (sequence / "endoscope2" / "depthL").mkdir(parents=True)
    (sequence / "endoscope2" / "toolL").mkdir(parents=True)
    (sequence / "K.txt").write_text("K placeholder\n")
    (sequence / "pose.txt").write_text("pose placeholder\n")
    for index in range(frames):
        name = f"frame_{index:06d}"
        Image.new("RGB", (8, 6), color=(index, 2, 3)).save(
            sequence / "endoscope2" / "L" / f"{name}.png"
        )
        np.save(sequence / "endoscope2" / "depthL" / f"{name}.npy", np.ones((3, 4)))
    return sequence


class SubmissionContractTests(unittest.TestCase):
    def test_new_view_is_callable(self) -> None:
        self.assertTrue(callable(new_view))

    def test_stage_never_links_public_target_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _make_sequence(root / "input")
            public_target = source / "endoscope1" / "L"
            public_target.mkdir(parents=True)
            Image.new("RGB", (8, 6), color=(255, 0, 0)).save(public_target / "secret.png")

            staged = root / "stage"
            _stage_source_only_sequence(source, staged)

            self.assertEqual(
                (staged / "endoscope1" / "L").resolve(),
                (source / "endoscope2" / "L").resolve(),
            )
            self.assertNotEqual(
                (staged / "endoscope1" / "L").resolve(),
                public_target.resolve(),
            )

    def test_validate_exact_rgb_png_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            sequence = _make_sequence(input_dir)
            renders = output_dir / sequence.name / "renders"
            renders.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (4, 3), color=(1, index, 3)).save(
                    renders / f"{index:05d}.png"
                )

            self.assertEqual([sequence], _sequence_dirs(input_dir))
            validate_output_tree(input_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
