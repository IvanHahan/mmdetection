import numpy as np
from stereo_utils import detection_utils
from stereo_utils.annots import create_labelme_annot
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
    focused = db["focused-elements"]
    inferencer = DetInferencer(
        "outputs/grounding_dino_swin-t.py",
        weights="outputs/iter_10000.pth",
    )

    query_labels = [
        "focused_element",
    ]

    for recording in tqdm(recordings.find({'screen_type': 'SAP'}, sort=[('_id', 1)])):

        if focused.find_one({'_id': recording['_id']}) is not None:
            continue
        image = image_from_base64(recording["image"])
        filename = recording["_id"]

        predictions = inferencer(
            np.array(image), 
            texts=". ".join(query_labels), 
            return_datasamples=True
        )["predictions"][0].pred_instances
        classes = inferencer.model.dataset_meta.get('classes', None)
        thresh = 0.3
        if predictions["scores"].max() < thresh:
            continue
        pred_boxes, new_labels = list(
            zip(
                *[
                    (b, l)
                    for b, l, s in zip(
                        predictions["bboxes"].cpu().numpy(),
                        predictions["label_names"] if 'label_names' in predictions else predictions['labels'].cpu().tolist(),
                        predictions["scores"],
                    )
                    if s > thresh
                ]
            )
        )
        
        raw_annot = create_labelme_annot(filename, image, pred_boxes, new_labels)

        os.makedirs(output, exist_ok=True)
        write_json(raw_annot, os.path.join(output, filename + ".json"))
        image.save(os.path.join(output, filename + ".jpg"))
