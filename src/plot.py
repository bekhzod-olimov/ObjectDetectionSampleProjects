import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from src.utils import makedirs

class YOLOVisualizer:
    def __init__(self, vis_dir, save_dir):
        
        self.vis_dir   = vis_dir
        self.save_dir  = save_dir
        self.save_name = os.path.basename(vis_dir)        

    def visualize(self):
        
        vis_files = glob(f"{self.vis_dir}/*.png")

        if not vis_files:
            print("No images found in the directory.")
            return

        for idx, vis_file in enumerate(vis_files):            
            title = os.path.splitext(os.path.basename(vis_file))[0]            
            img = Image.open(vis_file)
            plt.figure(figsize=(20, 10))  
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{title.upper()}", fontsize = 16)
            plt.savefig(f"{self.save_dir}/{self.save_name}_{title}.png")
            