import argparse
import json
import os

def prepare_coco_dataset(data_dir, output_dir):
    coco_anns = []
    coco_imgs = []
    class_mapping = {}
    class_hierarchy = {}
    img_id = 0
    ann_id = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    anns = json.load(f)
                    for ann in anns:
                        if ann["category"] not in class_mapping:
                            class_mapping[ann["category"]] = len(class_mapping)
                            class_hierarchy[class_mapping[ann["category"]]] = ann["supercategory"]
                        coco_anns.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_mapping[ann["category"]],
                            "segmentation": ann["segmentation"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "iscrowd": 0
                        })
                        ann_id += 1
                coco_imgs.append({
                    "id": img_id,
                    "file_name": ann["file_name"],
                    "height": ann["height"],
                    "width": ann["width"]
                })
                img_id += 1

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump({"annotations": coco_anns, "images": coco_imgs, "categories": [{"id": i, "name": name} for name, i in class_mapping.items()]}, f)
    with open(os.path.join(output_dir, "class_hierarchy.json"), "w") as f:
        json.dump(class_hierarchy, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a COCO format segmentation dataset for training a HierarchicalMaskRCNN")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared COCO dataset")
    args = parser.parse_args()

    prepare_coco_dataset(args.data_dir, args.output_dir)
