"""Render only the requested target RGB frames for an iMED NVS submission."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic

import mmcv
import torch
import torchvision

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams


def main() -> None:
    parser = ArgumentParser(description="Render iMED hidden target views")
    model_group = ModelParams(parser, sentinel=True)
    pipeline_group = PipelineParams(parser)
    hidden_group = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=2500)
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = get_combined_args(parser)
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    safe_state(True, seed=args.seed)
    dataset = model_group.extract(args)
    pipeline = pipeline_group.extract(args)
    hidden = hidden_group.extract(args)

    renders = args.output_dir / "renders"
    renders.mkdir(parents=True, exist_ok=False)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hidden)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
            load_test_cameras=True,
        )
        background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

        views = scene.getTestCameras()
        if len(views) == 0:
            raise ValueError("No target camera frames were constructed")
        torch.cuda.synchronize()
        started = monotonic()
        for index, view in enumerate(views):
            image = torch.clamp(
                render(view, gaussians, pipeline, background, mode="test")["render"],
                0.0,
                1.0,
            )
            torchvision.utils.save_image(image, renders / f"{index:05d}.png")
        torch.cuda.synchronize()
        elapsed = monotonic() - started
        print(f"Rendered {len(views)} target frames; FPS={len(views) / elapsed:.4f}")


if __name__ == "__main__":
    main()
