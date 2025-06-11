import os, cv2, random, numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from src.utils import makedirs

class YOLOv11Inference:
    def __init__(self, model, train_name, device, save_dir = None):
        
        self.res_dir, self.train_name = save_dir, train_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale, self.thickness = 0.3, 1
        if self.res_dir: makedirs(self.res_dir)
        self.device = device
        self.model = model.to(self.device)

    def run(self, image_dir, n_ims=15, rows=3):
        
        # Run inference        
        inference_results = self.model(image_dir, device=self.device, verbose=False)
        
        # Visualize results
        self.inference_vis(inference_results, n_ims, rows)      
            
    def demo(self, im_path):
        
        if isinstance(im_path, str):
            with Image.open(im_path) as im: 
                im = im.convert("RGB")
        else: im = Image.fromarray(im_path).convert("RGB")
        res = self.model(im, verbose = False)

        detection = self.process_res(r = res, or_im_rgb=np.array(im), demo=True)
                
        di = {}
        di["pred"] = detection        
        di["n_bboxes"] = self.num_bboxes
        di["original_im"] = im 
        
        return di
    
    def process_res(self, r, or_im_rgb, demo=None):

        self.num_bboxes = len(r)        
        for i in r:   
            # cls_names = i.names
            cls_names = {i: "DIQQAT" for i in range(len(i.names))}            
            for bbox in i.boxes:                
                # text = f"{list(cls_names.values())[int(bbox.cls.item())]} {(bbox.conf.item() * 100):.2f}%"
                text = f"{list(cls_names.values())[int(bbox.cls.item())]}"
                box = bbox.xyxy[0]
                x1, y1, x2, y2 = box
                coord1, coord2 = (int(x1), int(y1)), (int(x2), int(y2))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(or_im_rgb, coord1, coord2, color=color, thickness=3)
                (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
                text_x = coord1[0] + (coord2[0] - coord1[0] - text_width) // 2
                text_y = coord1[1] + (coord2[1] - coord1[1] + text_height) // 2
                cv2.putText(or_im_rgb, text, (text_x, text_y), self.font, self.font_scale, color, self.thickness, cv2.LINE_AA)
        
        if demo: return or_im_rgb

    def inference_vis(self, res, n_ims, rows):
        
        cols = n_ims // rows        
        plt.figure(figsize=(20, 10))

        for idx, r in enumerate(res):
            if idx == n_ims:
                break
            plt.subplot(rows, cols, idx + 1)
            or_im_rgb = np.array(Image.open(r.path).convert("RGB"))
            self.process_res(r = r, or_im_rgb=or_im_rgb)
            plt.imshow(or_im_rgb)            
            cnt_objects = "objects are" if self.num_bboxes >= 2 else "object is"
            plt.title(f"{self.num_bboxes} {cnt_objects} detected.")
            plt.axis("off")

        plt.savefig(os.path.join(self.res_dir, f"{self.train_name}_results.png"))