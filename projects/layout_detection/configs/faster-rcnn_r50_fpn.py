_base_ = [
    "base/faster-rcnn_r50_fpn.py",
    "base/dataset.py",
    "base/schedule.py",
    "base/runtime.py",
]

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
_base_.optim_wrapper.type = "AmpOptimWrapper"

load_from = "models/fasterrcnn/iter_5000.pth"
