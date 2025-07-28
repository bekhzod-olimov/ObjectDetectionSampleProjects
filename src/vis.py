import pickle, json, random
import os, cv2, yaml, numpy as np
from glob import glob
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, root, data_types, n_ims, rows, vis_dir, ds_nomi, cls_root, cmap=None, ann_type="coco"):
        self.root = root if ann_type == "yolo" else f"{os.path.dirname(os.path.dirname(os.path.dirname(root)))}_COCO"
        self.cls_root = cls_root
        os.makedirs(self.cls_root, exist_ok=True)
        self.n_ims, self.rows = n_ims, rows
        self.cmap, self.data_types = cmap, data_types
        self.vis_dir, self.ds_nomi = vis_dir, ds_nomi
        self.ann_type = ann_type
        os.makedirs(self.vis_dir, exist_ok=True)
        self.colors = ["firebrick", "darkorange", "blueviolet"]
        self.get_cls_names()
        self.get_bboxes()

    def get_cls_names(self):
        if self.ann_type == "yolo":
            with open(f"{self.root}/data.yaml", 'r') as file:
                data = yaml.safe_load(file)
            self.class_names = data['names']
        elif self.ann_type == "coco":
            for dt in self.data_types:
                print(f"self.root -> {self.root}")
                ann_path = f"{self.root}/annotations/{dt}_annotations.json"
                if os.path.isfile(ann_path):
                    with open(ann_path, 'r') as f:
                        data = json.load(f)
                    # categories may not be sorted by id, so sort!
                    cats = sorted(data['categories'], key=lambda x: x['id'])
                    self.class_names = [cat['name'] for cat in cats]
                    break
            else:
                raise RuntimeError("No COCO annotations files found for class names extraction.")
        self.class_dict = {index: name for index, name in enumerate(self.class_names)}
        with open(f"{self.cls_root}/{self.ds_nomi}_cls_names.pkl", "wb") as f: pickle.dump(self.class_names, f)
        print(f"Datasetdagi klasslar -> {self.class_names}")

    def get_bboxes(self):
        self.vis_datas, self.analysis_datas, self.im_paths = {}, {}, {}
        for data_type in self.data_types:
            all_bboxes, all_analysis_datas = [], {}
            if self.ann_type == "yolo":
                im_paths = glob(f"{self.root}/{data_type}/images/*")
                for im_path in im_paths:
                    bboxes = []
                    im_ext = os.path.splitext(im_path)[-1]
                    lbl_path = im_path.replace(im_ext, ".txt").replace(f"{data_type}/images", f"{data_type}/labels")
                    if not os.path.isfile(lbl_path): continue
                    for data in open(lbl_path):
                        parts = data.strip().split()[:5]
                        cls_index = int(parts[0])
                        cls_name = self.class_dict[cls_index]
                        bboxes.append([cls_name] + [float(x) for x in parts[1:]])
                        all_analysis_datas[cls_name] = all_analysis_datas.get(cls_name, 0) + 1
                    all_bboxes.append(bboxes)
                self.vis_datas[data_type] = all_bboxes
                self.analysis_datas[data_type] = all_analysis_datas
                self.im_paths[data_type] = im_paths
            elif self.ann_type == "coco":
                # Use COCO annotation JSON
                ann_path = f"{self.root}/annotations/{data_type}_annotations.json"
                with open(ann_path, "r") as f:
                    coco_ann = json.load(f)

                # Initialize required mappings and lists
                im_map = {}                 # image_id -> file_name
                coco_bboxes = {}            # image_id -> list of [class_name, x_min, y_min, w, h]
                all_analysis_datas = {}     # class_name -> count

                # Build map: image_id -> file_name
                images = coco_ann["images"]
                for img in images:
                    im_map[img["id"]] = img["file_name"]

                # Parse annotations
                annotations = coco_ann["annotations"]
                for ann in annotations:
                    image_id = ann["image_id"]
                    class_idx = ann["category_id"] - 1  # 0-based index
                    cls_name = self.class_dict[class_idx]
                    x_min, y_min, w, h = ann["bbox"]
                    bbox = [cls_name, x_min, y_min, w, h]
                    coco_bboxes.setdefault(image_id, []).append(bbox)
                    all_analysis_datas[cls_name] = all_analysis_datas.get(cls_name, 0) + 1

                # Rebuild image paths and all_bboxes in the order of the COCO 'images' array
                im_paths = []
                all_bboxes = []
                for img in images:
                    im_paths.append(img["file_name"])
                    all_bboxes.append(coco_bboxes.get(img["id"], []))

                self.vis_datas[data_type] = all_bboxes
                self.analysis_datas[data_type] = all_analysis_datas
                self.im_paths[data_type] = im_paths

    
    def plot(self, rows, cols, count, im_path, bboxes):
        plt.subplot(rows, cols, count)
        or_im = np.array(Image.open(im_path).convert("RGB"))
        height, width = or_im.shape[:2]        
        overlay = or_im.copy()

        for bbox in bboxes:
            if self.ann_type == "yolo":
                class_name, x_center, y_center, w, h = bbox
                # coords are normalized [0,1]; convert to px
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)
            else:
                # COCO format: [class_name, x_min, y_min, w, h] all px
                class_name, x_min, y_min, w, h = bbox
                x_min, y_min, w, h = float(x_min), float(y_min), float(w), float(h)
                x_max = int(x_min + w)
                y_max = int(y_min + h)
                x_min = int(x_min)
                y_min = int(y_min)
            # Draw rectangle (color generation can be improved)
            color = tuple([random.randint(0, 255) for _ in range(3)])
            # Ensure rectangle is within the image bounds:
            x_min, x_max = max(0, x_min), min(width-1, x_max)
            y_min, y_max = max(0, y_min), min(height-1, y_max)
            overlay = cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, thickness=3)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"There is (are) {len(bboxes)} object(s) in the image.")
        return count + 1

    def vis(self, save_name):
        print(f"{save_name.upper()} Data Visualization is in process...\n")
        assert self.cmap in ["rgb", "gray"], "Please choose rgb or gray cmap"
        cols = self.n_ims // self.rows; count = 1
        plt.figure(figsize=(25, 20))        
        indices = [random.randint(0, len(self.vis_datas[save_name]) - 1) for _ in range(self.n_ims)]
        for idx, index in enumerate(indices):
            if count == self.n_ims + 1: break
            im_path = self.im_paths[save_name][index]
            bboxes = self.vis_datas[save_name][index]
            count = self.plot(self.rows, cols, count, im_path=im_path, bboxes=bboxes)
        plt.savefig(f"{self.vis_dir}/{self.ds_nomi}_{save_name}_data_vis.png")

    def data_analysis(self, save_name, color):
        print("Data analysis is in process...\n")
        width, text_width, text_height = 0.7, 0.05, 2
        cls_names = list(self.analysis_datas[save_name].keys()); counts = list(self.analysis_datas[save_name].values())
        _, ax = plt.subplots(figsize=(30, 10))
        indices = np.arange(len(counts))
        ax.bar(indices, counts, width, color=color)
        ax.set_xlabel("Class Names", color="black")
        ax.set_xticks(range(len(cls_names)))
        ax.set_xticklabels(cls_names, rotation=90)
        ax.set(xticks=indices, xticklabels=cls_names)
        ax.set_ylabel("Data Counts", color="black")
        ax.set_title(f"{save_name.upper()} Dataset Class Imbalance Analysis")
        for i, v in enumerate(counts): ax.text(i - text_width, v + text_height, str(v), color="royalblue")
        plt.savefig(f"{self.vis_dir}/{self.ds_nomi}_{save_name}_data_analysis.png")

    def visualization(self): [self.vis(save_name) for save_name in self.data_types]
    def analysis(self): [self.data_analysis(save_name, color) for (save_name, color) in zip(self.data_types, self.colors)]