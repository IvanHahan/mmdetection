_base_ = ['./base/detr_r50_8xb2-150e_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/conditional_detr/conditional-detr_r50_8xb2-50e_coco/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth'

model = dict(
    type='ConditionalDETR',
    num_queries=900,
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=False),
            cross_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=True))),
    bbox_head=dict(
        type='ConditionalDETRHead',
        loss_cls=dict(
            _delete_=True,
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])))
