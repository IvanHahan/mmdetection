_base_ = "base/grounding_dino_swin-t.py"

load_from = "models/dino/iter_25000.pth"


optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0),
        }
    ),
)

auto_scale_lr = dict(base_batch_size=16)
