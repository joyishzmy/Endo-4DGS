_base_ = "./imed_extent10_principal_stereo_appearance_g5_iter2500.py"

ModelParams = dict(
    imed_stereo_adaptive_gate=True,
    imed_stereo_adaptive_confidence_floor=0.80,
)
