_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    './dataset.py',
    './schedule.py', 
    './runtime.py'
]

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
_base_.optim_wrapper.type = 'AmpOptimWrapper'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth'
