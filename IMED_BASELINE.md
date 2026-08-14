# iMED baseline record

This repository snapshot is based on upstream commit
`d4127d1b8f588a2c0e69a9b6fd24a9fafa2fcced`.

## Validated environment

- Python 3.8.20
- PyTorch 2.0.0+cu118
- torchvision 0.15.1+cu118
- NumPy 1.24.4
- CUDA device selection: GPU 0 only
- Random seed: 6666

## Reproducibility fixes

- Scale the iMED intrinsics from RGB resolution to depth/render resolution when
  constructing the global reprojection mask used by `metrics.py`.
- Rebuild the iMED overlap mask for every evaluation instead of reusing a stale
  cached mask.
- Fall back to the configured `camera_extent` for static iMED cameras when the
  camera-derived normalization radius is zero.

The corrected global overlap mask covers approximately 41.40% of a 512x640
frame. The previous unscaled implementation covered approximately 10.36%.

## Reference experiment

- Sequence: `session_004_scene_2_tool_1`
- Configuration: `arguments/imed.py`
- Coarse iterations: 2000
- Fine iterations: 6000
- Configured camera extent: 10
- Selected checkpoint: fine iteration 4000

Metrics use the corrected overlap mask:

| Checkpoint | PSNR (higher) | SSIM (higher) | LPIPS (lower) |
| --- | ---: | ---: | ---: |
| 1000 | 20.3324 | 0.6569 | 0.10794 |
| 3000 | 20.3897 | 0.6488 | 0.10360 |
| 4000 | 20.4123 | 0.6481 | 0.10352 |
| 5000 | 20.3687 | 0.6473 | 0.10357 |
| 6000 | 20.3209 | 0.6456 | 0.10404 |

Training data, rendered images, checkpoints, logs, and Conda environments are
not included in version control.
