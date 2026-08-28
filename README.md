# Endo-4DGS for iMED Static Two-Camera Evaluation

This repository supports the iMED 2026 challenge subtask on deformable novel view synthesis, part of EndoVis 2026 at MICCAI 2026 (Strasbourg, France).

[[Challenge Website](https://imed-challenge.github.io/)] [[Participate](https://www.synapse.org/Synapse:syn74277461/wiki/639538)] [[Parent Challenge Hub](https://opencas.dkfz.de/endovis/challenges/2026/)]

<p align="center">
  <img src="assets/comparison.gif" width="900" alt="Ground truth (left), trained endoscope render (middle), and evaluated endoscope render (right) on session_004_scene_2_tool_1"/>
</p>

## Huge Thanks to the Original Repository

This project is a fork of the original Endo-4DGS repository by Huang et al.

- Original codebase: [lastbasket/Endo-4DGS](https://github.com/lastbasket/Endo-4DGS)
- This fork: [smbonilla/Endo-4DGS](https://github.com/smbonilla/Endo-4DGS)

Please cite the original Endo-4DGS work:

```bibtex
@inproceedings{huang2024endo,
  title={Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={197--207},
  year={2024},
  organization={Springer}
}
```

## What This Fork Adds

This fork adds iMED dataset support for a static two-camera protocol:

- Training view: `endoscope2/L`
- Test view: `endoscope1/L`
- Depths: `depthL` (mm, used directly)
- Masks: `toolL`, loaded and converted as `mask = 1 - raw_mask/255`
- Poses: `pose.txt` with exactly 2 rows (`k tx ty tz qx qy qz qw`)
- Intrinsics: `K.txt`

Expected dataset layout:

```text
./data/imed/session_004_scene_2_tool_1
├── K.txt
├── pose.txt
├── endoscope1
│   ├── L
│   ├── depthL
│   └── toolL
└── endoscope2
    ├── L
    ├── depthL
    └── toolL
```

## Method: Train on One Camera, Test on the Other

For each static iMED session:

1. Train Gaussian splats only on `endoscope2` images.
2. Keep both cameras static and use the provided two-camera pose relation.
3. Render novel views from the held-out `endoscope1` camera.
4. Evaluate rendered test images against `endoscope1` ground truth.

This setup measures cross-camera generalization rather than interpolation within one camera stream.

## Metrics and Timing

- **PSNR / SSIM:** computed in `metrics.py`.
- **LPIPS:** also computed in `metrics.py`.
- **Inference speed:** printed as `FPS` in `render.py` during rendering.

For iMED two-camera evaluation, PSNR/SSIM are computed on the valid reprojection region from `endoscope2` into `endoscope1` (single global mask per sequence).

### Source-only Endoscope-2 stereo experiment

The optional stereo path uses only synchronized `endoscope2/L`, `endoscope2/R`,
`endoscope2/depthL`, `endoscope2/toolL`, `endoscope2/toolR`, `K2_L`, and `K2_R`
during training. It never reads Endoscope-1 images. Because the package does not
provide an explicit Endoscope-2 L/R extrinsic, first run
`scripts/estimate_imed_stereo_extrinsics.py`. The resulting calibration is stored
outside the original dataset under `calibration/imed/`; training refuses files
that do not certify source-only provenance or fail geometric validation.

`scripts/run_imed_stereo_rgbd_pilot.sh` performs a paired-seed two-sequence pilot:
the frozen gate090 baseline, robust RGB-D/multi-keyframe initialization (`G1`),
and synchronized L/R supervision with projected right pseudo-depth (`G2`).

The corrected v2 experiment is `scripts/run_imed_stereo_principal_v2.sh`. It
uses the actual off-center principal points from `K.txt` for every iMED camera.
Its right camera contributes RGB supervision only; projected left depth defines
right-view visibility but is excluded from depth, normal, smoothness, and depth
confidence losses. The script reruns a principal-point baseline and G1 control,
so results are paired rather than compared against legacy centered projections.

The G3 pilot is `scripts/run_imed_stereo_photogate_g3.sh`. It keeps the full
left RGB-D objective and treats the synchronized right view only as a detached,
exposure-aligned, photometrically gated auxiliary RGB loss. Right-view depth,
normal, confidence, and smoothness supervision remain disabled. It reuses the
seed-1 `baseline_pp` and `g2v2_rgb` controls from the v2 pilot.

The G4 pilot is `scripts/run_imed_stereo_fineonly_g4.sh`. It keeps G3 fixed but
removes the right camera entirely during the coarse stage, then enables the
same gated right RGB auxiliary only in the fine stage. This isolates whether
early stereo appearance gradients disturb geometry and densification.

The G5 pilot is `scripts/run_imed_stereo_appearance_g5.sh`. It keeps G4's stage
schedule, but the right-view auxiliary can backpropagate only to the Gaussian
SH features. Position, scale, rotation, opacity, deformation/grid parameters,
and densification statistics remain controlled exclusively by the left view.

## Setup

```bash
git clone https://github.com/smbonilla/Endo-4DGS.git
cd Endo-4DGS
git submodule update --init --recursive
conda create -n ED4DGS python=3.8
conda activate ED4DGS
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics
```

## Run iMED Training

```bash
sh train_imed.sh
```

## Run Rendering + Evaluation

```bash
python render.py --model_path <OUTPUT_PATH> --pc --skip_video --skip_train --configs arguments/imed.py
python metrics.py --model_path <OUTPUT_PATH>
```
