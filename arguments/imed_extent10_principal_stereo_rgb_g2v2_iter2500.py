_base_ = "./imed_extent10_principal_rgbd_g1_iter2500.py"

ModelParams = dict(
    imed_use_stereo=True,
    imed_stereo_calibration_dir="calibration/imed",
    imed_stereo_right_rgb_only=True,
)

OptimizationParams = dict(
    batch_size=2,
)
