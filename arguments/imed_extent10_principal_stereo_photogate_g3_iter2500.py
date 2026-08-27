_base_ = "./imed_extent10_principal_stereo_rgb_g2v2_iter2500.py"

ModelParams = dict(
    imed_stereo_photometric_gate=True,
    imed_stereo_photometric_sigma=0.10,
    imed_stereo_right_loss_weight=0.25,
)
