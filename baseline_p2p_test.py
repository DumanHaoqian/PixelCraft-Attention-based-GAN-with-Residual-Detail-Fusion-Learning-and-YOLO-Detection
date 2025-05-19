#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pix2Pix: Test code for Semantic-to-Real translation
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from typing import Union
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from p2p import UNetGenerator
# --------------------------------------------------------------------------
# Test Configuration
# --------------------------------------------------------------------------
class TestConfig:
    data_root = "test_dataset"  # Should contain input_test and target_test folders
    result_dir = "test_results"
    ckpt_path = "checkpoints/pix2pix_epoch_latest.pth"
    batch_size = 8
    num_workers = 4
    img_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_cfg = TestConfig()
Path(test_cfg.result_dir).mkdir(exist_ok=True, parents=True)

# --------------------------------------------------------------------------
# Test Dataset
# --------------------------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root: Union[str, Path], img_size: int = 256):
        root = Path(root)
        self.rootA = root / "input_test"
        self.rootB = root / "target_test"
        
        self.paths = sorted(list(self.rootA.glob("*")))
        assert len(self.paths) > 0, f"未在 {self.rootA} 找到任何文件！"
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1,1]
        ])
    
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        pathA = self.paths[idx]
        pathB = self.rootB / pathA.name
        imgA = Image.open(pathA).convert("RGB")
        imgB = Image.open(pathB).convert("RGB")
        return {
            "A": self.transform(imgA),
            "B": self.transform(imgB),
            "name": pathA.stem
        }

# --------------------------------------------------------------------------
# Test Function
# --------------------------------------------------------------------------
def compute_metrics(real, fake):
    # tensor [-1,1] → numpy [0,1]
    real = ((real.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
    fake = ((fake.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
    psnr_list, ssim_list = [], []
    for r, f in zip(real, fake):
        psnr_list.append(psnr(r, f, data_range=1))

        try:
            ssim_list.append(ssim(r, f, channel_axis=-1, data_range=1))
        except TypeError:
            ssim_list.append(ssim(r, f, multichannel=True, data_range=1))
    return np.mean(psnr_list), np.mean(ssim_list)

def test():
    # Load the model
    print(f": {test_cfg.ckpt_path}")
    ckpt = torch.load(test_cfg.ckpt_path, map_location=test_cfg.device)
    
    # Initialize the model
    netG = UNetGenerator().to(test_cfg.device)
    netG.load_state_dict(ckpt["G"])
    netG.eval()
    
    # Create test dataloader
    test_dataset = TestDataset(test_cfg.data_root, test_cfg.img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_cfg.batch_size,
        shuffle=False,
        num_workers=test_cfg.num_workers,
        pin_memory=True
    )
    print(f": {len(test_dataset)}")
    
    # Run test
    psnr_total, ssim_total, n_samples = 0, 0, 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            real_A = data["A"].to(test_cfg.device)
            real_B = data["B"].to(test_cfg.device)
            names = data["name"]
            
            # Generate fake images
            fake_B = netG(real_A)
            
            # Calculate metrics
            p, s = compute_metrics(real_B, fake_B)
            batch_size = real_A.size(0)
            psnr_total += p * batch_size
            ssim_total += s * batch_size
            n_samples += batch_size
            
            # Save results
            for j in range(batch_size):
                img_name = names[j]
                # Save generated image
                vutils.save_image(
                    (fake_B[j] + 1) / 2,  # Convert from [-1,1] to [0,1]
                    f"{test_cfg.result_dir}/{img_name}_generated.png"
                )
                
                # Save comparison grid (input, generated, target)
                grid = vutils.make_grid(
                    [real_A[j], fake_B[j], real_B[j]],
                    nrow=3, normalize=True, scale_each=True
                )
                vutils.save_image(
                    grid,
                    f"{test_cfg.result_dir}/{img_name}_comparison.png"
                )
            
            print(f"Batch {i+1}/{len(test_loader)} processed")
    
    # Calculate averages
    avg_psnr = psnr_total / n_samples
    avg_ssim = ssim_total / n_samples
    
    # Print and save results
    print("\nOutput:")
    print(f"avg PSNR: {avg_psnr:.2f} dB")
    print(f"avg SSIM: {avg_ssim:.4f}")
    
    with open(f"{test_cfg.result_dir}/metrics.txt", "w") as f:
        f.write("Test output:\n")
        f.write(f"avg PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"image number: {n_samples}\n")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix")
    parser.add_argument("--data_root", default=test_cfg.data_root,
                        help="")
    parser.add_argument("--result_dir", default=test_cfg.result_dir,
                        help="")
    parser.add_argument("--ckpt_path", default=test_cfg.ckpt_path,
                        help="")
    parser.add_argument("--batch_size", type=int, default=test_cfg.batch_size,
                        help="")
    
    args = parser.parse_args()
    
    test_cfg.data_root = args.data_root
    test_cfg.result_dir = args.result_dir
    test_cfg.ckpt_path = args.ckpt_path
    test_cfg.batch_size = args.batch_size
    
    test()