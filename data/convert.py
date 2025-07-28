import os
import json
import yaml
from pathlib import Path
from typing import List, Dict
from PIL import Image
import shutil
from datetime import date

class YOLOtoCOCOConverter:
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = Path(root_dir)
        self.data_yaml_path = Path(os.path.join(root_dir, "data.yaml"))
        self.output_dir = Path(output_dir)
        self.class_names = self.get_class_names_from_yaml()

    def get_class_names_from_yaml(self) -> List[str]:
        with open(self.data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def find_split_dirs(self) -> Dict[str, Dict[str, Path]]:
        splits = {}
        split_map = {'train': 'train', 'valid': 'valid', 'test': 'test'}
        for orig, new in split_map.items():
            split_dir = self.root_dir / orig
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            if images_dir.is_dir() and labels_dir.is_dir():
                splits[new] = {'images': images_dir, 'labels': labels_dir}
        return splits

    def convert_split(self, split_name: str, images_dir: Path, labels_dir: Path):
        # Prepare output dirs
        save_images_dir = self.output_dir / split_name
        save_ann_dir = self.output_dir / "annotations"
        os.makedirs(save_images_dir, exist_ok=True)
        os.makedirs(save_ann_dir, exist_ok=True)

        # COCO meta information
        coco = {
            "info": {
                "description": "Dataset in COCO format",
                "version": "1.0",
                "year": date.today().year,
                "contributor": "Automatically converted",
                "date_created": str(date.today())
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ],
            "type": "instances",
            "images": [],
            "annotations": [],
            "categories": []
        }
        # Categories
        for idx, class_name in enumerate(self.class_names, 1):
            coco["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "none"
            })

        annotation_id = 1
        image_map = {}  # avoid duplicate images if YOLO does duplicate labels!

        for label_file in sorted(labels_dir.glob("*.txt")):
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                image_name = label_file.stem + ext
                image_path = images_dir / image_name
                if image_path.is_file():
                    break
            else:
                continue

            # Copy image to output split
            save_path = save_images_dir / image_name
            shutil.copy(image_path, save_path)

            # COCO image info
            with Image.open(image_path) as img:
                width, height = img.size

            image_id = len(image_map) + 1
            rel_image_path = str(save_path.resolve())
            image_map[image_name] = image_id

            coco["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": rel_image_path,
                "date_captured": date.today().year
            })

            # YOLO label parsing to COCO annotation
            with open(label_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = parts
                class_id = int(class_id) + 1
                x_center = float(x_center) * width
                y_center = float(y_center) * height
                w = float(w) * width
                h = float(h) * height
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                bbox = [x_min, y_min, w, h]
                area = w * h
                # Rectangle segmentation: [x0, y0, x1, y0, x1, y1, x0, y1]
                seg = [[
                    x_min, y_min,
                    x_min + w, y_min,
                    x_min + w, y_min + h,
                    x_min, y_min + h
                ]]
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": seg
                }
                coco["annotations"].append(annotation)
                annotation_id += 1

        # Save COCO JSON
        ann_file = save_ann_dir / f"{split_name}_annotations.json"
        with open(ann_file, "w") as f:
            json.dump(coco, f, indent=4)
        print(f"{split_name}: COCO annotation saved to {ann_file}")

    def convert_all(self):
        splits = self.find_split_dirs()
        for split_name, dirs in splits.items():
            self.convert_split(split_name, dirs['images'], dirs['labels'])