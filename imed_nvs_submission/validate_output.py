"""Validate the official iMED NVS output tree and RGB PNG contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _expected_frame_count(sequence_dir: Path) -> int:
    paths = sorted((sequence_dir / "endoscope2" / "L").glob("*.png"))
    if not paths:
        raise ValueError(f"No source RGB frames in {sequence_dir}")
    return len(paths)


def _expected_resolution(sequence_dir: Path) -> tuple[int, int]:
    depth_paths = sorted((sequence_dir / "endoscope2" / "depthL").glob("*.npy"))
    if not depth_paths:
        raise ValueError(f"No source depth frames in {sequence_dir}")
    height, width = np.load(depth_paths[0], mmap_mode="r").shape
    return int(width), int(height)


def validate_sequence_output(sequence_dir: Path, output_dir: Path) -> None:
    render_dir = output_dir / "renders"
    if not render_dir.is_dir():
        raise ValueError(f"Missing renders directory: {render_dir}")

    expected_count = _expected_frame_count(sequence_dir)
    expected_names = {f"{index:05d}.png" for index in range(expected_count)}
    actual_files = [path for path in render_dir.iterdir() if path.is_file()]
    actual_names = {path.name for path in actual_files}
    if actual_names != expected_names:
        raise ValueError(
            f"{render_dir}: missing={sorted(expected_names - actual_names)[:5]}, "
            f"extra={sorted(actual_names - expected_names)[:5]}"
        )

    expected_size = _expected_resolution(sequence_dir)
    for path in sorted(actual_files):
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"Not a PNG image: {path}")
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image at {path}, got {image.mode}")
            if image.size != expected_size:
                raise ValueError(
                    f"Resolution mismatch at {path}: {image.size} != {expected_size}"
                )


def validate_output_tree(input_dir: Path, output_dir: Path) -> None:
    from imed_nvs_submission.predict import _sequence_dirs

    sequences = _sequence_dirs(input_dir)
    expected_names = {sequence.name for sequence in sequences}
    actual_names = {path.name for path in output_dir.iterdir() if path.is_dir()}
    if actual_names != expected_names:
        raise ValueError(
            f"Output sequence directories differ: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for sequence in sequences:
        validate_sequence_output(sequence, output_dir / sequence.name)
    print(f"Output contract validation passed for {len(sequences)} sequence(s).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    validate_output_tree(args.input_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
