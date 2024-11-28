import os

# dataset settings
dataset_type = "CocoDataset"

train_annots = "focused_elements_coco/train.json"
val_annots = "focused_elements_coco/val.json"
img_prefix = "focused_elements_labelme/"
image_size = (1600, 900)
data_root = (
    os.environ.get("INPUT_DATA", "/home/azureuser/cloudfiles/code/datasets") + "/"
)

class_name = (
    "focused_element",
    # "layout",
    # "title",
    # "checkbox",
    # "section_title",
    # "button",
    # "active_tab",
    # "tab",
    # "value",
    # "website",
    # "dropdown",
    # "selected_value",
    # "selected_checkbox",
    # "icon",
)
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (120, 120, 220),
        (180, 120, 120),
        (6, 230, 230),
        (80, 50, 50),
        (4, 200, 3),
        (120, 120, 80),
        (140, 140, 190),
        (204, 5, 255),
        (230, 230, 230),
        (4, 250, 7),
        (224, 5, 255),
        (235, 255, 7),
        (150, 5, 61),
        (120, 120, 70),
        (8, 255, 51),
        (255, 6, 82),
        (143, 255, 140),
        (204, 255, 4),
        (255, 51, 7),
        (204, 70, 3),
        (0, 102, 200),
        (61, 230, 250),
        (255, 6, 51),
        (11, 102, 255),
        (255, 7, 71),
    ],
)

backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Color", prob=0.6),
    dict(type="Invert", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [dict(type="RandomChoiceResize", scales=[image_size], keep_ratio=True)],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(1024, 768), (2048, 1024), (2536, 1024)],
                    keep_ratio=False,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(512, 512),
                    allow_negative_crop=True,
                ),
                dict(type="RandomChoiceResize", scales=[image_size], keep_ratio=True),
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(1024, 768), (2048, 1024), (2536, 1024)],
                    keep_ratio=False,
                ),
                dict(type="RandomChoiceResize", scales=[image_size], keep_ratio=True),
            ],
        ],
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "text",
            "custom_entities",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="FixScaleResize", scale=image_size, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "text",
            "custom_entities",
        ),
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_annots,
        data_prefix=dict(img=img_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        return_classes=True,
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_annots,
        return_classes=True,
        data_prefix=dict(img=img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + val_annots,
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator
