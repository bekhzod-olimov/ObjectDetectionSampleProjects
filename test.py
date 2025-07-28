# import subprocess
# import os

# class DFINETrainer:
#     def __init__(self, config_path, save_dir, model_ckpt=None, device="cuda:0", nproc=4, additional_args=None):
#         self.config_path = config_path
#         self.save_dir = save_dir
#         self.model_ckpt = model_ckpt
#         self.device = device
#         self.nproc = nproc
#         self.additional_args = additional_args or []

#     def train(self, use_amp=True, seed=0):
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if self.device.startswith("cuda") else self.device

#         cmd = [
#             "torchrun", "--master_port=7777",
#             f"--nproc_per_node={self.nproc}",
#             "train.py",
#             "-c", self.config_path,
#         ]
#         if self.model_ckpt is not None:
#             cmd += ["-t", self.model_ckpt]
#         if use_amp:
#             cmd.append("--use-amp")
#         cmd.extend(["--seed", str(seed)])
#         cmd.extend(self.additional_args)

#         subprocess.run(cmd, cwd=self.save_dir, env=env)

# trainer = DFINETrainer(
#     config_path="configs/dfine/custom/dfine_hgnetv2_l_custom.yml",
#     save_dir="D-FINE",
#     model_ckpt="/home/bekhzod/Desktop/backup/dfine/dfine_l_coco.pth",               # Optional
#     device="0",                           # CUDA device number as str
#     nproc=1
# )
# trainer.train()

import os

# Usage Example:

