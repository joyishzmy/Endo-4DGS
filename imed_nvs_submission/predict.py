"""Official iMED NVS Docker entry point.

The evaluator mounts hidden source data read-only at ``/input`` and a writable
directory at ``/output``.  This module intentionally never reads target-view
Endoscope-1 RGB, depth, or masks.  A source-backed placeholder target stream is
created under ``/tmp`` only to instantiate camera timestamps and dimensions;
the target pose and intrinsics still come from ``pose.txt`` and ``K.txt``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


APP_ROOT = Path(os.environ.get("IMED_APP_ROOT", "/app"))
DEFAULT_CONFIG = APP_ROOT / "arguments/imed_extent10_smooth002_softoverlap_iter2500.py"
REQUIRED_SOURCE_PATHS = (
    "K.txt",
    "pose.txt",
    "endoscope2/L",
    "endoscope2/depthL",
)


def _sequence_dirs(input_dir: Path) -> list[Path]:
    sequences = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_dir():
            continue
        missing = [relative for relative in REQUIRED_SOURCE_PATHS if not (path / relative).exists()]
        if missing:
            print(f"Skipping non-sequence directory {path.name}: missing {missing}")
            continue
        sequences.append(path)
    if not sequences:
        raise ValueError(f"No valid iMED NVS sequences found under {input_dir}")
    return sequences


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _stage_source_only_sequence(source: Path, staged: Path) -> None:
    """Create a writable source-only view of one sequence under ``/tmp``."""

    staged.mkdir(parents=True, exist_ok=False)
    _symlink(source / "K.txt", staged / "K.txt")
    _symlink(source / "pose.txt", staged / "pose.txt")

    source_scope = source / "endoscope2"
    staged_source_scope = staged / "endoscope2"
    staged_source_scope.mkdir()
    _symlink(source_scope / "L", staged_source_scope / "L")
    _symlink(source_scope / "depthL", staged_source_scope / "depthL")
    if (source_scope / "toolL").is_dir():
        _symlink(source_scope / "toolL", staged_source_scope / "toolL")
    else:
        (staged_source_scope / "toolL").mkdir()

    # The original renderer expects a test-camera stream.  Link only source
    # files, never public/hidden target files, then apply K1_L and camera-1 pose
    # when the loader constructs these cameras.
    staged_target_scope = staged / "endoscope1"
    staged_target_scope.mkdir()
    _symlink(source_scope / "L", staged_target_scope / "L")
    _symlink(source_scope / "depthL", staged_target_scope / "depthL")
    if (source_scope / "toolL").is_dir():
        _symlink(source_scope / "toolL", staged_target_scope / "toolL")
    else:
        (staged_target_scope / "toolL").mkdir()


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=APP_ROOT, env=env, check=True)


def _run_sequence(
    source: Path,
    output_dir: Path,
    config: Path,
    seed: int,
    port: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"imed_nvs_{source.name}_") as temporary:
        work = Path(temporary)
        staged = work / "data" / source.name
        model = work / "model"
        _stage_source_only_sequence(source, staged)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        env["PYTHONHASHSEED"] = str(seed)
        env.setdefault("OMP_NUM_THREADS", "4")

        _run(
            [
                sys.executable,
                "train.py",
                "-s",
                str(staged),
                "-m",
                str(model),
                "--configs",
                str(config),
                "--seed",
                str(seed),
                "--port",
                str(port),
                "--test_iterations",
                "-1",
                "--save_iterations",
                "2500",
            ],
            env,
        )

        sequence_output = output_dir / source.name
        _run(
            [
                sys.executable,
                "-m",
                "imed_nvs_submission.render_target",
                "-s",
                str(staged),
                "-m",
                str(model),
                "--configs",
                str(config),
                "--iteration",
                "2500",
                "--output-dir",
                str(sequence_output),
                "--seed",
                str(seed),
            ],
            env,
        )

        from imed_nvs_submission.validate_output import validate_sequence_output

        validate_sequence_output(source, sequence_output)


def new_view(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> None:
    """Train on Endoscope 2 and render the held-out Endoscope-1 view."""

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    config = Path(os.environ.get("IMED_CONFIG", str(DEFAULT_CONFIG))).resolve()
    seed = int(os.environ.get("IMED_SEED", "1"))

    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    if not config.is_file():
        raise ValueError(f"Submission config does not exist: {config}")
    output_path.mkdir(parents=True, exist_ok=True)

    sequences = _sequence_dirs(input_path)
    started = time.monotonic()
    print(
        f"iMED NVS: sequences={len(sequences)}, seed={seed}, config={config.name}",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    for index, sequence in enumerate(sequences):
        print(f"\n===== [{index + 1}/{len(sequences)}] {sequence.name} =====", flush=True)
        try:
            _run_sequence(sequence, output_path, config, seed, 6100 + index)
        except Exception as error:  # preserve scores for all other sequences
            failures.append((sequence.name, str(error)))
            shutil.rmtree(output_path / sequence.name, ignore_errors=True)
            print(f"ERROR: {sequence.name}: {error}", file=sys.stderr, flush=True)

    elapsed = time.monotonic() - started
    print(f"iMED NVS completed in {elapsed / 60.0:.2f} minutes", flush=True)
    if failures:
        details = "; ".join(f"{name}: {message}" for name, message in failures)
        raise RuntimeError(f"{len(failures)} sequence(s) failed: {details}")

    from imed_nvs_submission.validate_output import validate_output_tree

    validate_output_tree(input_path, output_path)


def main() -> None:
    new_view(
        os.environ.get("INPUT_DIR", "/input"),
        os.environ.get("OUTPUT_DIR", "/output"),
    )


if __name__ == "__main__":
    main()
