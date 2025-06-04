import os, cv2, random, numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from src.utils import makedirs

class YOLOv11Inference:
    def __init__(self, model, train_name, save_dir = None, device="cuda"):
        
        self.res_dir, self.train_name = save_dir, train_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale, self.thickness = 1, 2        
        if self.res_dir: makedirs(self.res_dir)
        self.device = device
        self.model = model.to(self.device)

    def run(self, image_dir, n_ims=15, rows=3):
        
        # Run inference        
        inference_results = self.model(image_dir, device=self.device, verbose=False)
        
        # Visualize results
        self.inference_vis(inference_results, n_ims, rows)

    def demo(self, im_path):
        
        with Image.open(im_path) as im: im = im.convert("RGB")
        im_size = im.size[0], im.size[1]                 
        res = self.model(im_path, verbose = False)

        detection = self.process_res(r = res, or_im_rgb=np.array(im), demo=True)
                
        di = {}
        di["pred"] = detection        
        di["n_bboxes"] = self.num_bboxes
        di["original_im"] = im 
        
        return di
    
    def process_res(self, r, or_im_rgb, demo=None):

        self.num_bboxes = len(r)
        print(f"self.num_bboxes -> {self.num_bboxes}")
        for i in r:   
            cls_names = i.names                       
            for bbox in i.boxes:                
                text = f"{list(cls_names.values())[int(bbox.cls.item())]} {(bbox.conf.item() * 100):.2f}%"
                box = bbox.xyxy[0]
                x1, y1, x2, y2 = box
                coord1, coord2 = (int(x1), int(y1)), (int(x2), int(y2))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(or_im_rgb, coord1, coord2, color=color, thickness=10)
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

# class Denormalize:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
        
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#         return tensor

# class ModelInferenceVisualizer:
#     def __init__(self, model, device, mean, std, outputs_dir, ds_nomi, class_names=None, im_size=224):
        
#         self.denormalize = Denormalize(mean, std)
#         self.model = model
#         self.device = device
#         self.class_names = class_names
#         self.outputs_dir = outputs_dir        
#         self.ds_nomi = ds_nomi
#         self.im_size = im_size
#         self.model.eval()  
#         # self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(class_names)).to(self.device)

#     def tensor_to_image(self, tensor):
        
#         tensor = self.denormalize(tensor)  
#         tensor = tensor.permute(1, 2, 0)  
#         return (tensor.cpu().numpy() * 255).astype(np.uint8)

#     def plot_value_array(self, logits, gt, class_names):
        
#         probs = torch.nn.functional.softmax(logits, dim=1)
#         pred_class = torch.argmax(probs, dim=1)
        
#         plt.grid(visible=True)
#         plt.xticks(range(len(class_names)), class_names, rotation='vertical')
#         plt.yticks(np.arange(0.0, 1.1, 0.1))
#         bars = plt.bar(range(len(class_names)), [p.item() for p in probs[0]], color="#777777")
#         plt.ylim([0, 1])
#         if isinstance(gt, str):
#             bars[pred_class].set_color('green') if pred_class.item() == class_names[gt] else bars[pred_class].set_color('red')            
#         else:
#             bars[pred_class].set_color('green') if pred_class.item() == gt else bars[pred_class].set_color('red')
        
#         # Save figure to a buffer
#         import io
#         buf = io.BytesIO()
#         plt.tight_layout()
#         plt.savefig(buf, format='png')
#         plt.close()  # Close the figure after saving it to free memory
#         buf.seek(0)

#         return buf 

#     def generate_cam_visualization(self, image_tensor):
        
#         cam = GradCAM(model=self.model, target_layers=[self.model.features[-1].conv], use_cuda=self.device == "cuda")
#         grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0, :]
#         return grayscale_cam

#     def demo(self, im_path):
        
#         with Image.open(im_path) as im: im = im.convert("RGB")
#         im_size = im.size[0], im.size[1] 
#         if isinstance(im_path, str): gt = os.path.splitext(os.path.basename(im_path))[0].split("___")[-1]
#         else: gt = "Uploaded Image"         
#         im_tn = get_tfs()(im).unsqueeze(dim = 0).to(self.device)
#         with torch.no_grad(): logits = self.model(im_tn)
#         pred   = torch.argmax(logits, dim=1)

#         grayscale_cam = cv2.resize(self.generate_cam_visualization(im_tn.squeeze(dim=0)), im_size)
#         gradcam = show_cam_on_image(np.array(im).astype(np.uint8) / 255, grayscale_cam, image_weight=0.4, use_rgb=True)
#         if logits.dim() == 1:  # If 1D, add a batch dimension
#             logits = logits.unsqueeze(0)
                
#         di = {}
#         di["pred"] = pred.item()
#         di["gt"] = gt
#         di["original_im"] = im
#         di["gradcam"] = gradcam
#         di["probs"]   = self.plot_value_array(logits=logits, gt=gt, class_names=self.class_names)
#         di["confidence"] = (torch.max(torch.nn.functional.softmax(logits, dim=1), dim = 1)[0].item() * 100)
        
#         return di

#     def infer_and_visualize(self, test_dl, num_images=5, rows=2, demo=False):
        
#         preds, images, lbls, logitss = [], [], [], []
#         accuracy, count = 0, 1
#         with torch.no_grad():

#             for idx, batch in tqdm(enumerate(test_dl), desc="Inference"):
#                 im, gt = TrainValidation.to_device(batch, device=self.device)                
#                 logits = self.model(im)
#                 pred_class = torch.argmax(logits, dim=1)
                
#                 accuracy += (pred_class == gt).sum().item()
#                 self.f1_metric.update(logits, gt)  
        
#                 images.append(im[0])
#                 logitss.append(logits[0])
#                 preds.append(pred_class[0].item())
#                 lbls.append(gt[0].item())
        
#         # Compute metrics AFTER the loop
#         print(f"Accuracy of the model on the test data -> {(accuracy / len(test_dl.dataset)):.3f}")
#         print(f"F1 score of the model on the test data -> {(self.f1_metric.compute().item()):.3f}") 

#         plt.figure(figsize=(20, 10))
#         indices = [random.randint(0, len(images) - 1) for _ in range(num_images)]
#         for idx, index in enumerate(indices):
#             # Convert and denormalize image
#             im = self.tensor_to_image(images[index].squeeze())
#             pred_idx = preds[index]
#             gt_idx = lbls[index]

#             # Display image
#             plt.subplot(rows, 2 * num_images // rows, count)
#             count += 1
#             plt.imshow(im, cmap="gray")
#             plt.axis("off")

#             # GradCAM visualization
#             grayscale_cam = self.generate_cam_visualization(images[index])
#             visualization = show_cam_on_image(im / 255, grayscale_cam, image_weight=0.4, use_rgb=True)
#             plt.imshow(cv2.resize(visualization, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet')
#             plt.axis("off")

#             # Prediction probability array
#             logits = logitss[index]
#             if logits.dim() == 1:  # If 1D, add a batch dimension
#                 logits = logits.unsqueeze(0)
#             plt.subplot(rows, 2 * num_images // rows, count)
#             count += 1
#             bars = self.plot_value_array(logits=logits, gt=gt_idx, class_names=self.class_names)

#             # Title with GT and Prediction
#             if self.class_names:
#                 gt_name = self.class_names[gt_idx]
#                 pred_name = self.class_names[pred_idx]
#                 color = "green" if gt_name == pred_name else "red"
#                 plt.title(f"GT -> {gt_name} ; PRED -> {pred_name}", color=color)
        
#         os.makedirs(self.outputs_dir, exist_ok = True)
#         plt.savefig(f"{self.outputs_dir}/{self.ds_nomi}_model_performance_analysis.png")

#         # Plot confusion matrix
#         plt.figure(figsize=(20, 10))
#         cm = confusion_matrix(lbls, preds)
#         sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.savefig(f"{self.outputs_dir}/{self.ds_nomi}_confusion_matrix.png")