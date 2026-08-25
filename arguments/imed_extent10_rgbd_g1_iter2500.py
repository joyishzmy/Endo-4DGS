_base_ = "./imed_extent10_smooth002_softoverlap_gate090_iter2500.py"

ModelParams = dict(
    imed_metric_depth_loss=True,
    imed_pretrain_keyframes=3,
    imed_pretrain_max_points=360000,
)
