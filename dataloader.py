import os
import torch
from torch.utils.data import Dataset
import json
from pycocotools.coco import COCO

class HierarchicalCOCO(Dataset):
    def __init__(self, root_dir, ann_file, hierarchy_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        self.img_ids = self.coco.getImgIds()
        with open(hierarchy_file) as f:
            self.hierarchy = json.load(f)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = self.coco.annToMask(anns[0])
        for ann in anns[1:]:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        if self.transform:
            img = self.transform(img)
        return (img, mask, self.hierarchy[str(anns[0]['category_id'])])

