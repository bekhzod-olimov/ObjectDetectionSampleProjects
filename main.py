import os, yaml
import torch
import argparse
import timm
import pickle
from src.train import YOLOv11Trainer
from src.vis import Visualization
from src.plot import YOLOVisualizer
from src.infer import YOLOv11Inference
from src.transform import get_tfs
from data.fetch import DatasetDownloader
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model")

    parser.add_argument('--dataset_name', type=str, default="malaria", help="Name of the dataset")    
    parser.add_argument('--dataset_root', type=str, default="/home/bekhzod/Desktop/backup/object_detection_project_datasets", help="Root folder for datasets")
    parser.add_argument('--device', type=str, default="cuda", help="Root folder for class names")
    parser.add_argument('--cls_root', type=str, default="saved_cls_names", help="Root folder for class names")
    parser.add_argument('--vis_dir', type=str, default="vis", help="Directory for visualizations")
    parser.add_argument('--learning_curve_dir', type=str, default="learning_curves", help="Directory to save learning curves")
    parser.add_argument('--outputs_dir', type=str, default="results/images", help="Directory for inference results")
    parser.add_argument('--model_name', type=str, default="yolo11n.pt", help="Model architecture from Ultralytics")    
    parser.add_argument('--image_size', type=int, default=480, help="Input image size for the model")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for dataloaders")
    parser.add_argument('--epochs', type=int, default=20, help="Early stopping patience")

    return parser.parse_args()

def main():
    args = parse_args()    
    # ds_nomlari = ["covid", "malaria"]
    # ds_nomlari = ["military"]    
    # ds_nomlari = ["covid"]    

    # for ds_nomi in ds_nomlari:

    assert args.device in ["cpu", "cuda"], "Please choose correct device to run the detection model!"
    # print(f"{ds_nomi} dataset bilan train jarayoni boshlanmoqda...")

    # args.dataset_name = ds_nomi  
    device = device if args.device == "cpu" else [0]
    ds_nomi = args.dataset_name    

    if ds_nomi in ["baggage", "fish"]:          
        ds_path = os.path.join(args.dataset_root, args.dataset_name, args.dataset_name)        
    elif ds_nomi == "military":
        ds_path = os.path.join(args.dataset_root, args.dataset_name, args.dataset_name, args.dataset_name, "KIIT-MiTA")
        yml_file_path = f"{ds_path}/KIIT-MiTA.yml"

        with open(yml_file_path, 'r') as file: data = yaml.safe_load(file)
            
        # Update the paths
        data['train'] = f'{ds_path}/train/images'
        data['val'] =   f'{ds_path}/valid/images'
        data['test'] = f'{ds_path}//test/images'
        
        # Save the updated YAML content to a file
        output_path = f'{ds_path}/data.yaml'            
        with open(output_path, 'w') as file: yaml.dump(data, file, default_flow_style=False)
    
    train_name = f"{ds_nomi}_{os.path.splitext(args.model_name)[0]}"
    save_path = os.path.join("runs", "detect", train_name)

    if not os.path.isdir(ds_path): DatasetDownloader(save_dir=ds_path).download(ds_nomi=args.dataset_name)
    else: print(f"{args.dataset_name} dataseti allaqachon {args.dataset_root} yo'lagiga yuklab olingan.")             

    vis = Visualization(root = ds_path, data_types = ["train", "valid", "test"], n_ims = 20, rows = 5, 
                        vis_dir = args.vis_dir, ds_nomi = ds_nomi, cmap = "rgb")
    vis.analysis(); vis.visualization()

    os.makedirs(args.cls_root, exist_ok=True)
    with open(f"{args.cls_root}/{args.dataset_name}_cls_names.pkl", "wb") as f: pickle.dump(vis.class_names, f)

    print(f"Datasetdagi klasslar -> {vis.class_names}")

    trainer = YOLOv11Trainer(model_path=os.path.join("ckpts", args.model_name), data_yaml=os.path.join(ds_path, "data.yaml"), 
                                train_name=train_name, device=args.device)
    trainer.train(epochs = args.epochs, imgsz=args.image_size)   

    print(f"\nTraining process is completed. Visualizing learning curves...")     
    visualizer = YOLOVisualizer(vis_dir=save_path, save_dir=args.learning_curve_dir)
    visualizer.visualize()        

    print(f"\nInference process is going to start with the pre-trained model...")
    
    model = YOLO(os.path.join(save_path, "weights", "best.pt"))
    yolo_infer = YOLOv11Inference(model, save_dir=args.outputs_dir, train_name=train_name, device=args.device)
    yolo_infer.run(image_dir = f"{ds_path}/test/images")

if __name__ == "__main__": main()