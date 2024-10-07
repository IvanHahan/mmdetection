_base_ = "./conditional-detr_r50_8xb2-50e_coco.py"

data_root = (
    "/home/azureuser/cloudfiles/code/datasets/screen_obj_detection/"  # dataset root
)

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 200
stage2_num_epochs = 1
base_lr = 0.00002

metainfo = {
    "classes": [
        "button",
        "tabbar",
        "value",
        "title",
        "text",
        "dropdown",
        "stabbar",
        "section",
        "icon",
        "checkbox",
        "scheckbox",
    ],
    "palette": [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
        (165, 42, 42),
        (255, 77, 255),
        (0, 226, 252),
        (182, 182, 255),
        (0, 82, 0),
        (120, 166, 157),
        (110, 76, 0),
        (174, 57, 255),
        (199, 100, 0),
        (72, 0, 118),
        (255, 179, 240),
        (0, 125, 92),
        (209, 0, 151),
        (188, 208, 182),
        (0, 220, 176),
        (255, 99, 164),
        (92, 0, 73),
        (133, 129, 255),
        (78, 180, 255),
        (0, 228, 0),
        (174, 255, 243),
        (45, 89, 255),
        (134, 134, 103),
        (145, 148, 174),
        (255, 208, 186),
        (197, 226, 255),
        (171, 134, 1),
        (109, 63, 54),
        (207, 138, 255),
        (151, 0, 95),
        (9, 80, 61),
        (84, 105, 51),
        (74, 65, 105),
        (166, 196, 102),
        (208, 195, 210),
        (255, 109, 65),
        (0, 143, 149),
        (179, 0, 194),
        (209, 99, 106),
        (5, 121, 0),
        (227, 255, 205),
        (147, 186, 208),
        (153, 69, 1),
        (3, 95, 161),
        (163, 255, 0),
        (119, 0, 170),
        (0, 182, 199),
        (0, 165, 120),
        (183, 130, 88),
        (95, 32, 0),
        (130, 114, 135),
        (110, 129, 133),
        (166, 74, 118),
        (219, 142, 185),
        (79, 210, 114),
        (178, 90, 62),
        (65, 70, 15),
        (127, 167, 115),
        (59, 105, 106),
        (142, 108, 45),
        (196, 172, 0),
        (95, 54, 80),
        (128, 76, 255),
        (201, 57, 1),
        (246, 0, 122),
        (191, 162, 208),
    ],
}


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img="train/"),
        ann_file="train_refined.json",
        metainfo=metainfo,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img="train/"),
        ann_file="train_refined.json",
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "train_refined.json")

test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=len(metainfo["classes"])))

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize", scale=(640, 640), ratio_range=(0.1, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=(640, 640)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=2, save_best="auto"  # only keep latest 2 checkpoints
    ),
    logger=dict(type="LoggerHook", interval=5),
)

custom_hooks = [
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    )
]

# load COCO pre-trained weight
load_from = "./work_dirs/obj_detection_config/epoch_175.pth"

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
visualizer = dict(
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
)
