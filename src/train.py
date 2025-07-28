import subprocess
import os, yaml, torch

class YOLOv11Trainer:
    def __init__(self, model_path, data_yaml, train_name, device):

        from ultralytics import YOLO
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.train_name = train_name

    def train(self, epochs=1, imgsz=480):
        
        self.model.train(
            data=self.data_yaml,
            name=self.train_name,
            epochs=epochs,
            imgsz=imgsz,
            device=self.device
        )    


class DFINETrainer:
    def __init__(self, config_path, save_dir, model_ckpt=None, device="cuda:0", nproc=4, additional_args=None):
        self.config_path = config_path
        self.save_dir = save_dir
        self.model_ckpt = model_ckpt
        self.device = device
        self.nproc = nproc
        self.additional_args = additional_args or []

    
    def edit_config(self, train_img_folder, train_ann_file, val_img_folder, val_ann_file, output_config_path=None):
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Edit train dataloader paths
        cfg['train_dataloader']['dataset']['img_folder'] = train_img_folder
        cfg['train_dataloader']['dataset']['ann_file'] = train_ann_file
        # Edit val dataloader paths
        cfg['val_dataloader']['dataset']['img_folder'] = val_img_folder
        cfg['val_dataloader']['dataset']['ann_file'] = val_ann_file

        # Save to file (overwrite or new)
        output_path = output_config_path if output_config_path else self.config_path
        with open(output_path, 'w') as f:
            yaml.safe_dump(cfg, f)

        # Update path for training
        self.config_path = output_path

    def train(self, use_amp=True, seed=0):        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if self.device.startswith("cuda") else self.device

        cmd = [
            "torchrun", "--master_port=7777",
            f"--nproc_per_node={self.nproc}",
            "train.py",
            "-c", self.config_path,
        ]
        if self.model_ckpt is not None:
            cmd += ["-t", self.model_ckpt]
        if use_amp:
            cmd.append("--use-amp")
        cmd.extend(["--seed", str(seed)])
        cmd.extend(self.additional_args)

        subprocess.run(cmd, cwd=self.save_dir, env=env)

