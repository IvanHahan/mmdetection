import numpy as np
from stereo_utils import detection_utils
from stereo_utils.annots import boxes_to_labelme_shapes
from stereo_utils.common import image_from_base64, image_to_base64, write_json
import os
from stereo_utils.mongo import records_db
from tqdm import tqdm

from mmdet.apis import DetInferencer

if __name__ == "__main__":
    # output = "/home/azureuser/cloudfiles/code/datasets/focused_elements_labelme"
    output = './focused_elements'
    db = records_db()
    recordings = db["snapshots"]
    annots = db["eltc"]
    screen_layout = db["screen_layout"]
    inferencer = DetInferencer(
        "models/dino/grounding_dino_swin-t.py",
        weights="models/dino/iter_25000.pth",
    )

    category_map = dict(
        section="section_title",
        stab="active_tab",
        stabbar='active_tab',
        svalue="selected_value",
        scheckbox="selected_checkbox",
        column="value",
    )

    query_labels = [
        "layout",
    ]

    for raw_annot in tqdm(annots.find({"type": {"$ne": "funsd"}})):

        recording = recordings.find_one(
            {"_id": raw_annot["_id"], "screen_type": {"$ne": "funsd"}}
        )
        if recording is None:
            continue

        image = image_from_base64(recording["image"])
        filename = recording["_id"]

        del raw_annot["_id"]
        filtered_shapes = []
        for shape in raw_annot["shapes"]:
            if shape["shape_type"] != "rectangle":
                continue
            if len(shape["points"]) == 4:
                shape["points"] = [
                    [shape["points"][0], shape["points"][1]],
                    [shape["points"][2], shape["points"][3]],
                ]
            shape["label"] = category_map.get(shape["label"], shape["label"])
            if shape["label"] in [
                # "section_title",
                # "title",
                # "value",
                # "button",
                "active_tab",
                # "tab",
                # "website",
                "selected_value",
                # "checkbox",
                "selected_checkbox",
                # "dropdown",
                # "icon",
            ]:
                shape['label'] = 'focused'
                filtered_shapes.append(shape)

        if len(filtered_shapes) == 0:
            continue
        annot_boxes = np.array([s["points"] for s in filtered_shapes]).reshape(-1, 4)
        # predictions = inferencer(
        #     np.array(image), 
        #     texts=". ".join(query_labels), 
        #     return_datasamples=True
        # )["predictions"][0].pred_instances
        # classes = inferencer.model.dataset_meta.get('classes', None)
        # pred_boxes, new_labels = list(
        #     zip(
        #         *[
        #             (b, l)
        #             for b, l, s in zip(
        #                 predictions["bboxes"].cpu().numpy(),
        #                 predictions["label_names"] if 'label_names' in predictions else predictions['labels'].cpu().tolist(),
        #                 predictions["scores"],
        #             )
        #             if s > 0.3
        #         ]
        #     )
        # )
        # if isinstance(new_labels[0], int):
        #     new_labels = [classes[l] for l in new_labels]

        # new_labels = np.array(new_labels)
        # pred2annot = detection_utils.match_rectangles(pred_boxes, annot_boxes)


        # for i, annot2pred in enumerate(pred2annot.T):
        #     new_labels[annot2pred == 1] = filtered_shapes[i]["label"]

        # new_shapes = boxes_to_labelme_shapes(np.array(pred_boxes), new_labels)

        raw_annot["flags"] = {}
        raw_annot["version"] = "5.3.1"
        raw_annot["imageHeight"] = image.size[1]
        raw_annot["imageWidth"] = image.size[0]
        raw_annot["imageData"] = image_to_base64(image)
        raw_annot["imagePath"] = filename + ".jpg"
        raw_annot["shapes"] = filtered_shapes

        # screen_layout.update_one({"_id": filename}, {"$set": raw_annot}, upsert=True)
        os.makedirs(output, exist_ok=True)
        write_json(raw_annot, os.path.join(output, filename + ".json"))
        image.save(os.path.join(output, filename + ".jpg"))
