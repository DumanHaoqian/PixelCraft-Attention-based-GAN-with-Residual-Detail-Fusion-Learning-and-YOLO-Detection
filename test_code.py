import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from attenDetailGan import UNetGeneratorPlus,DetailNet,FusionModel
# Test Configuration
class TestConfig:
    data_root = "test_dataset"    # Path to test dataset folder
    result_dir = "test_results"   # Where to save results
    ckpt_path = "checkpoints/pix2pix_plus_detail_latest.pth"
    batch_size = 8
    num_workers = 4
    img_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_cfg = TestConfig()
Path(test_cfg.result_dir).mkdir(parents=True, exist_ok=True)

# Dataset for testing (similar to training dataset structure)
class TestDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.root = Path(root)
        self.input_dir = self.root / "input_test"
        self.target_dir = self.root / "target_test"
        
        self.input_paths = sorted(list(self.input_dir.glob("*.*")))
        assert len(self.input_paths) > 0, f"No images found in {self.input_dir}"
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3)
        ])
    
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        target_path = self.target_dir / input_path.name
        
        return {
            "A": self.transform(Image.open(input_path).convert("RGB")),
            "B": self.transform(Image.open(target_path).convert("RGB")),
            "name": input_path.stem
        }
def psnr_ssim(real, fake):
    r=((real.cpu()+1)/2).numpy().transpose(0,2,3,1)
    f=((fake.cpu()+1)/2).numpy().transpose(0,2,3,1)
    ps, ss = [], []
    for rr, ff in zip(r, f):
        ps.append(psnr(rr, ff, data_range=1))
        try: ss.append(ssim(rr, ff, channel_axis=-1, data_range=1))
        except TypeError: ss.append(ssim(rr, ff, multichannel=True, data_range=1))
    return np.mean(ps), np.mean(ss)
def test():
    # Load model checkpoint
    print(f"Loading checkpoint from {test_cfg.ckpt_path}")
    ckpt = torch.load(test_cfg.ckpt_path, map_location=test_cfg.device)
    
    # Initialize models
    netG = UNetGeneratorPlus().to(test_cfg.device)
    detailNet = DetailNet().to(test_cfg.device)
    fusionModel = FusionModel(netG, detailNet).to(test_cfg.device)
    
    # Load weights
    netG.load_state_dict(ckpt["G"])
    detailNet.load_state_dict(ckpt["Detail"])
    fusionModel.alpha.data = ckpt["alpha"]
    print(f"Alpha value: {fusionModel.alpha.item():.3f}")
    
    # Create test dataloader
    test_dataset = TestDataset(test_cfg.data_root, test_cfg.img_size)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_cfg.batch_size,
        shuffle=False,
        num_workers=test_cfg.num_workers,
        pin_memory=True
    )
    print(f"Testing on {len(test_dataset)} images...")
    
    # Set models to evaluation mode
    netG.eval()
    detailNet.eval()
    
    # Initialize metrics
    ps_main, ss_main = 0, 0
    ps_detail, ss_detail = 0, 0
    ps_fusion, ss_fusion = 0, 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["A"].to(test_cfg.device)
            targets = batch["B"].to(test_cfg.device)
            names = batch["name"]
            
            # Generate outputs
            main_outputs = netG(inputs)
            detail_outputs = detailNet(inputs)
            fusion_outputs = fusionModel(inputs)
            
            # Calculate metrics
            p_main, s_main = psnr_ssim(targets, main_outputs)
            p_detail, s_detail = psnr_ssim(targets, detail_outputs)
            p_fusion, s_fusion = psnr_ssim(targets, fusion_outputs)
            
            # Update totals
            batch_size = inputs.size(0)
            ps_main += p_main * batch_size
            ss_main += s_main * batch_size
            ps_detail += p_detail * batch_size
            ss_detail += s_detail * batch_size
            ps_fusion += p_fusion * batch_size
            ss_fusion += s_fusion * batch_size
            total_samples += batch_size
            
            # Save output images
            for i in range(batch_size):
                img_name = names[i]
                vutils.save_image((fusion_outputs[i] + 1) / 2, 
                                 f"{test_cfg.result_dir}/{img_name}_fusion.png")
    
    # Calculate average metrics
    avg_psnr_main = ps_main / total_samples
    avg_ssim_main = ss_main / total_samples
    avg_psnr_detail = ps_detail / total_samples
    avg_ssim_detail = ss_detail / total_samples
    avg_psnr_fusion = ps_fusion / total_samples
    avg_ssim_fusion = ss_fusion / total_samples
    
    # Print results
    print("\nTest Results:")
    print(f"Main Generator:  PSNR = {avg_psnr_main:.2f}  SSIM = {avg_ssim_main:.4f}")
    print(f"Detail Network:  PSNR = {avg_psnr_detail:.2f}  SSIM = {avg_ssim_detail:.4f}")
    print(f"Fusion Model:    PSNR = {avg_psnr_fusion:.2f}  SSIM = {avg_ssim_fusion:.4f}")
    print(f"Alpha value: {fusionModel.alpha.item():.3f}")
    
    # Save metrics to file
    with open(f"{test_cfg.result_dir}/metrics.txt", "w") as f:
        f.write("Test Results:\n")
        f.write(f"Main Generator:  PSNR = {avg_psnr_main:.2f}  SSIM = {avg_ssim_main:.4f}\n")
        f.write(f"Detail Network:  PSNR = {avg_psnr_detail:.2f}  SSIM = {avg_ssim_detail:.4f}\n")
        f.write(f"Fusion Model:    PSNR = {avg_psnr_fusion:.2f}  SSIM = {avg_ssim_fusion:.4f}\n")
        f.write(f"Alpha value: {fusionModel.alpha.item():.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PixelCraft model")
    parser.add_argument("--data_root", default=test_cfg.data_root,
                       help="Path to test dataset folder")
    parser.add_argument("--result_dir", default=test_cfg.result_dir,
                       help="Directory to save test results")
    parser.add_argument("--ckpt_path", default=test_cfg.ckpt_path,
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    test_cfg.data_root = args.data_root
    test_cfg.result_dir = args.result_dir
    test_cfg.ckpt_path = args.ckpt_path
    
    test()