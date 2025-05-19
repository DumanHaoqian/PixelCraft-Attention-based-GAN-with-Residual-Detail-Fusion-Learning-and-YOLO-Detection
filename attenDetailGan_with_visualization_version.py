"""
===================================================================
PixelCraft (Improved U-Net + ResBlock + Self-Attention + DetailNet)
With Visualization
===================================================================
"""

import argparse, random
from pathlib import Path
from typing import Union
import os
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity  as ssim
from torch.utils.tensorboard import SummaryWriter

# ────────────────────────────────────────────────────
# 0. Comfigure
# ────────────────────────────────────────────────────
class Config:
    data_root   = "ds_copy" # The reference dataset          
    ckpt_dir    = "checkpoints"
    sample_dir  = "samples"
    log_dir     = "runs/pix2pix_plus_detail"
    
    # Visualization directories
    vis_dir     = "visualizations"
    loss_dir    = "visualizations/losses"
    attn_dir    = "visualizations/attention"
    output_dir  = "visualizations/outputs"
    comp_dir    = "visualizations/comparison"
    metrics_dir  = "visualizations/metrics"

    epochs      = 200
    batch_size  = 8
    num_workers = 4
    lr          = 2e-4
    beta1       = 0.5
    lambda_L1   = 30
    img_size    = 256
    resume      = "checkpoints"
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.sample_dir).mkdir(parents=True, exist_ok=True)

# Create visualization directories
Path(cfg.vis_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.loss_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.loss_dir + "/combined").mkdir(parents=True, exist_ok=True)
Path(cfg.loss_dir + "/individual").mkdir(parents=True, exist_ok=True)
Path(cfg.attn_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.comp_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.metrics_dir).mkdir(parents=True, exist_ok=True)
# ────────────────────────────────────────────────────
# 1. Dataset
# ────────────────────────────────────────────────────
class SemanticRealPair(Dataset):
    def __init__(self, root:Union[str,Path], phase:str="train", img_size:int=256):
        root = Path(root)
        if phase == "train":
            self.Adir, self.Bdir = root/"input_train", root/"target_train"
        else:
            self.Adir, self.Bdir = root/"input_val",   root/"target_val"
        self.paths = sorted(self.Adir.glob("*"))
        assert self.paths, f"Can not find images in {self.Adir} !"
        flip = phase == "train"
        self.tf = T.Compose([
            T.Resize((img_size, img_size), Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5 if flip else 0.0),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3)
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        pa = self.paths[idx]; pb = self.Bdir/pa.name
        return dict(
            A = self.tf(Image.open(pa).convert("RGB")),
            B = self.tf(Image.open(pb).convert("RGB")),
            name = pa.stem)

def make_loader(phase):
    ds = SemanticRealPair(cfg.data_root, phase, cfg.img_size)
    return DataLoader(ds, cfg.batch_size, shuffle=phase=="train",
                      num_workers=cfg.num_workers, pin_memory=True)

# ────────────────────────────────────────────────────
# 2. Network
# ────────────────────────────────────────────────────
def init_weights(m, gain=0.02):
    if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d,
                    nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight, 0.0, gain)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)

# 2-1 Base Model
class ConvINReLU(nn.Sequential):
    """Conv → (Optional)InstanceNorm → ReLU"""
    def __init__(self, inC, outC, k=3, s=1, p=1, use_norm=True):
        layers = [nn.Conv2d(inC, outC, k, s, p, bias=False)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(outC, affine=True))
        layers.append(nn.ReLU(True))
        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels))
    def forward(self, x): return x + self.block(x)

class SelfAttn(nn.Module):
    """SAGAN Self-Attention"""
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value = nn.Conv2d(in_dim, in_dim,     1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attention_map = None  # To store attention map for visualization
        
    def forward(self, x):
        B,C,H,W = x.size()
        q = self.query(x).view(B, -1, H*W)          # B,Cq,N
        k = self.key(x).view(B, -1, H*W)            # B,Ck,N
        v = self.value(x).view(B, -1, H*W)          # B,C ,N
        attn = torch.bmm(q.permute(0,2,1), k)       # B,N,N
        attn = F.softmax(attn/(C**0.5), dim=-1)
        out  = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)
        
        # Store attention map for visualization
        self.attention_map = attn.detach()
        
        return self.gamma*out + x

# 2-2 Generator
class UpBlock(nn.Module):
    def __init__(self, inC, outC, drop=False):
        super().__init__()
        layers = [nn.Upsample(scale_factor=2, mode="nearest"),
                  nn.Conv2d(inC, outC, 3, 1, 1, bias=False),
                  nn.InstanceNorm2d(outC, affine=True),
                  nn.ReLU(True)]
        if drop: layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UNetGeneratorPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        # ---------- Encoder ----------
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1),
                                nn.LeakyReLU(0.2, True))
        self.e2 = ConvINReLU(ngf,    ngf*2, 4, 2, 1)
        self.e3 = ConvINReLU(ngf*2,  ngf*4, 4, 2, 1)
        self.e4 = ConvINReLU(ngf*4,  ngf*8, 4, 2, 1)
        self.e5 = ConvINReLU(ngf*8,  ngf*8, 4, 2, 1)
        self.e6 = ConvINReLU(ngf*8,  ngf*8, 4, 2, 1)
        self.e7 = ConvINReLU(ngf*8,  ngf*8, 4, 2, 1)
        self.e8 = ConvINReLU(ngf*8,  ngf*8, 4, 2, 1, use_norm=False) # 1×1

        # ---------- Bottleneck ----------
        self.res = nn.Sequential(*(ResBlock(ngf*8) for _ in range(3)))

        # ---------- Self-Attention  ----------
        self.attn4 = SelfAttn(ngf * 16)   # d4 ⊕ e4  → 1024
        self.attn3 = SelfAttn(ngf * 8)    # d5 ⊕ e3  → 512
        self.attn2 = SelfAttn(ngf * 4)    # d6 ⊕ e2  → 256

        # ---------- Decoder ----------
        self.d1 = UpBlock(ngf*8,  ngf*8, drop=True)
        self.d2 = UpBlock(ngf*16, ngf*8, drop=True)
        self.d3 = UpBlock(ngf*16, ngf*8, drop=True)
        self.d4 = UpBlock(ngf*16, ngf*8)
        self.d5 = UpBlock(ngf*16, ngf*4)
        self.d6 = UpBlock(ngf*8,  ngf*2)
        self.d7 = UpBlock(ngf*4,  ngf)
        self.d8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf*2, out_ch, 3, 1, 1),
            nn.Tanh())

        self.apply(init_weights)
        
        # Store intermediate features for visualization
        self.encoder_features = {}
        self.decoder_features = {}

    def forward(self, x):
        # Encoder
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); e8=self.e8(e7)
        
        # Store encoder features
        self.encoder_features = {
            'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4,
            'e5': e5, 'e6': e6, 'e7': e7, 'e8': e8
        }
        
        # Bottleneck
        b = self.res(e8)
        
        # Decoder with skip connections
        d1 = self.d1(b);          d1 = torch.cat([d1, e7], 1)
        d2 = self.d2(d1);         d2 = torch.cat([d2, e6], 1)
        d3 = self.d3(d2);         d3 = torch.cat([d3, e5], 1)
        d4 = self.d4(d3);         d4_attn = self.attn4(torch.cat([d4, e4], 1))
        d5 = self.d5(d4_attn);    d5_attn = self.attn3(torch.cat([d5, e3], 1))
        d6 = self.d6(d5_attn);    d6_attn = self.attn2(torch.cat([d6, e2], 1))
        d7 = self.d7(d6_attn);    d7 = torch.cat([d7, e1], 1)
        out= self.d8(d7)
        
        # Store decoder features
        self.decoder_features = {
            'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4_attn,
            'd5': d5_attn, 'd6': d6_attn, 'd7': d7
        }
        
        return out

# 2-2a. DetailNet: Encode for details
class DetailNet(nn.Module):
    """DetailNet: Capture the high-frequency details lost by the main generator"""
    def __init__(self, in_ch=3, out_ch=3, nf=32):
        super().__init__()
        # lightweigh encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf*2, 4, 2, 1),
            nn.InstanceNorm2d(nf*2, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1),
            nn.InstanceNorm2d(nf*4, affine=True),
            nn.LeakyReLU(0.2, True),
        )
    
        # Residual Bolock Handling
        self.res1 = ResBlock(nf*4)
        self.res2 = ResBlock(nf*4)
        self.attn = SelfAttn(nf*4)  # Self-attention mechanism helps capture global information.
        
        # Decoder: Up sampling
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*4, nf*2, 3, 1, 1),
            nn.InstanceNorm2d(nf*2, affine=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*2, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf, affine=True),
            nn.ReLU(True),
            nn.Conv2d(nf, out_ch, 3, 1, 1),
            nn.Tanh()  # Control the output of Residual to [-1,1]
        )
        
        self.apply(init_weights)
        
        # Store features for visualization
        self.encoded_features = None
        self.res_features = None
    
    def forward(self, x):
        feat = self.enc(x)
        self.encoded_features = feat
        
        res1_out = self.res1(feat)
        res2_out = self.res2(res1_out)
        attn_out = self.attn(res2_out)
        self.res_features = attn_out
        
        out = self.dec(attn_out)
        return out

# 2-3 Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        def C(inC, outC, s=2):
            return nn.Sequential(
                nn.Conv2d(inC, outC, 4, s, 1, bias=False),
                nn.InstanceNorm2d(outC, affine=True),
                nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            C(ndf,   ndf*2),
            C(ndf*2, ndf*4),
            C(ndf*4, ndf*8, 1),
            nn.Conv2d(ndf*8, 1, 4, 1, 1))
        self.apply(init_weights)
    def forward(self, x): return self.net(x)

# 2-4 Model Fusion
class FusionModel(nn.Module):
    """Fusion the main Generator and DetailNet output"""
    def __init__(self, generator, detail_net, alpha=0.8):
        super().__init__()
        self.generator = generator
        self.detail_net = detail_net
        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable weight alpha

    def forward(self, x):
        main_output = self.generator(x)
        detail_output = self.detail_net(x)
        # Fusion output (Weighted sum)
        return torch.clamp(self.alpha * main_output + (1 - self.alpha) * detail_output, -1, 1)

# ────────────────────────────────────────────────────
# 3. Tools
# ────────────────────────────────────────────────────
def save_sample(ep, idx, rA, rB, fB):
    grid = vutils.make_grid(torch.cat([rA, fB, rB], 0),
                            nrow=rA.size(0), normalize=True, scale_each=True)
    vutils.save_image(grid, f"{cfg.sample_dir}/ep{ep:03d}_b{idx:04d}.png")

def save_ckpt(netG, netD, optG, optD, ep):
    torch.save(dict(G=netG.state_dict(), D=netD.state_dict(),
                    optG=optG.state_dict(), optD=optD.state_dict(),
                    epoch=ep),
               f"{cfg.ckpt_dir}/pix2pix_plus_{ep:03d}.pth")

def save_ckpt_extended(netG, netD, detailNet, alpha, optG, optD, optDetail, ep):
    torch.save(dict(G=netG.state_dict(),
                   D=netD.state_dict(),
                   Detail=detailNet.state_dict(),
                   alpha=alpha,
                   optG=optG.state_dict(),
                   optD=optD.state_dict(),
                   optDetail=optDetail.state_dict(),
                   epoch=ep),
              f"{cfg.ckpt_dir}/pix2pix_plus_detail_{ep:03d}.pth")

def psnr_ssim(real, fake):
    # 添加 detach() 确保没有梯度追踪
    r = ((real.detach().cpu() + 1) / 2).numpy().transpose(0, 2, 3, 1)
    f = ((fake.detach().cpu() + 1) / 2).numpy().transpose(0, 2, 3, 1)
    ps, ss = [], []
    for rr, ff in zip(r, f):
        ps.append(psnr(rr, ff, data_range=1))
        try: ss.append(ssim(rr, ff, channel_axis=-1, data_range=1))
        except TypeError: ss.append(ssim(rr, ff, multichannel=True, data_range=1))
    return np.mean(ps), np.mean(ss)

# ────────────────────────────────────────────────────
# 3.1 Visualization Tools
# ────────────────────────────────────────────────────
def plot_metrics(val_losses, epoch):
    """绘制PSNR和SSIM指标曲线"""
    # 绘制PSNR指标曲线
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses['PSNR'], label='Fusion', linewidth=2, color='blue')
    plt.plot(val_losses['PSNR_Main'], label='Main Generator', linewidth=2, color='green', linestyle='--')
    plt.plot(val_losses['PSNR_Detail'], label='DetailNet', linewidth=2, color='red', linestyle=':')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.metrics_dir}/psnr_ep{epoch:03d}.png", dpi=300)
    plt.close()
    
    # 绘制SSIM指标曲线
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses['SSIM'], label='Fusion', linewidth=2, color='blue')
    plt.plot(val_losses['SSIM_Main'], label='Main Generator', linewidth=2, color='green', linestyle='--')
    plt.plot(val_losses['SSIM_Detail'], label='DetailNet', linewidth=2, color='red', linestyle=':')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('SSIM Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.metrics_dir}/ssim_ep{epoch:03d}.png", dpi=300)
    plt.close()

def visualize_model_components_with_metrics(input_img, gt_img, main_output, detail_output, fusion_output, alpha_value, epoch, idx):
    """可视化模型各组件及其输出结果，同时显示PSNR和SSIM指标"""
    # 转换tensor为numpy数组以便可视化
    input_np = ((input_img.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    gt_np = ((gt_img.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    main_np = ((main_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    detail_np = ((detail_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    fusion_np = ((fusion_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    
    # 计算PSNR和SSIM值
    main_psnr, main_ssim = psnr_ssim(gt_img, main_output)
    detail_psnr, detail_ssim = psnr_ssim(gt_img, detail_output)
    fusion_psnr, fusion_ssim = psnr_ssim(gt_img, fusion_output)
    
    # 计算残差（细节增强部分）
    residual = (detail_output - main_output).detach().cpu()
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    residual_np = residual.permute(0, 2, 3, 1).numpy()
    
    # 选择要可视化的样本（批次中的第一个）
    sample_idx = 0
    
    # 创建综合可视化
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(input_np[sample_idx])
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(main_np[sample_idx])
    plt.title(f'Main Output\nPSNR: {main_psnr:.2f}, SSIM: {main_ssim:.3f}')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(detail_np[sample_idx])
    plt.title(f'DetailNet Output\nPSNR: {detail_psnr:.2f}, SSIM: {detail_ssim:.3f}')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(residual_np[sample_idx])
    plt.title('Detail Enhancement (Residual)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(fusion_np[sample_idx])
    plt.title(f'Fusion Output (α={alpha_value:.2f})\nPSNR: {fusion_psnr:.2f}, SSIM: {fusion_ssim:.3f}')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(gt_np[sample_idx])
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{cfg.comp_dir}/ep{epoch:03d}_b{idx:04d}_comp_metrics.png", dpi=300)
    plt.close()
    
    # 保存带度量指标的所有样本图像
    batch_size = input_img.size(0)
    for i in range(min(batch_size, 4)):  # 最多显示4个样本
        grid = vutils.make_grid(
            torch.cat([
                input_img[i:i+1].detach(), 
                main_output[i:i+1].detach(), 
                detail_output[i:i+1].detach(), 
                fusion_output[i:i+1].detach(), 
                gt_img[i:i+1].detach()
            ], 0),
            nrow=5, normalize=True, scale_each=True
        )
        vutils.save_image(grid, f"{cfg.output_dir}/ep{epoch:03d}_b{idx:04d}_sample{i}_metrics.png")
    
    return fusion_psnr, fusion_ssim, main_psnr, main_ssim, detail_psnr, detail_ssim

def visualize_validation_metrics(val_loader, netG, detailNet, fusion_model, epoch):
    """分析并可视化验证集上的图像质量指标"""
    all_fusion_psnr = []
    all_fusion_ssim = []
    all_main_psnr = []
    all_main_ssim = []
    all_detail_psnr = []
    all_detail_ssim = []
    
    with torch.no_grad():
        for i, b in enumerate(val_loader):
            A = b["A"].to(cfg.device)
            B = b["B"].to(cfg.device)
            
            # 前向传播
            fB_main = netG(A)
            fB_detail = detailNet(A)
            fB_fusion = fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail
            
            # 计算指标
            p_fusion, s_fusion = psnr_ssim(B, fB_fusion)
            p_main, s_main = psnr_ssim(B, fB_main)
            p_detail, s_detail = psnr_ssim(B, fB_detail)
            
            # 收集指标
            bs = A.size(0)
            all_fusion_psnr.extend([p_fusion] * bs)
            all_fusion_ssim.extend([s_fusion] * bs)
            all_main_psnr.extend([p_main] * bs)
            all_main_ssim.extend([s_main] * bs)
            all_detail_psnr.extend([p_detail] * bs)
            all_detail_ssim.extend([s_detail] * bs)
    
    # 创建PSNR vs SSIM散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(all_fusion_ssim, all_fusion_psnr, label='Fusion', alpha=0.7, s=50)
    plt.scatter(all_main_ssim, all_main_psnr, label='Main', alpha=0.7, s=50)
    plt.scatter(all_detail_ssim, all_detail_psnr, label='Detail', alpha=0.7, s=50)
    plt.xlabel('SSIM', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR vs SSIM for Different Components', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.metrics_dir}/psnr_vs_ssim_ep{epoch:03d}.png", dpi=300)
    plt.close()
    
    # 创建PSNR直方图
    plt.figure(figsize=(12, 6))
    plt.hist(all_fusion_psnr, bins=20, alpha=0.7, label='Fusion')
    plt.hist(all_main_psnr, bins=20, alpha=0.7, label='Main')
    plt.hist(all_detail_psnr, bins=20, alpha=0.7, label='Detail')
    plt.xlabel('PSNR (dB)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of PSNR Values', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.metrics_dir}/psnr_hist_ep{epoch:03d}.png", dpi=300)
    plt.close()
    
    # 创建SSIM直方图
    plt.figure(figsize=(12, 6))
    plt.hist(all_fusion_ssim, bins=20, alpha=0.7, label='Fusion')
    plt.hist(all_main_ssim, bins=20, alpha=0.7, label='Main')
    plt.hist(all_detail_ssim, bins=20, alpha=0.7, label='Detail')
    plt.xlabel('SSIM', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of SSIM Values', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.metrics_dir}/ssim_hist_ep{epoch:03d}.png", dpi=300)
    plt.close()

def plot_losses(train_losses, val_losses, epoch):
    """Plot and save loss curves"""
    # Create combined loss plot
    plt.figure(figsize=(14, 10))
    
    # Plot training losses
    for name, values in train_losses.items():
        plt.plot(values, label=f'Train {name}', linewidth=2)
    
    # Plot validation losses
    for name, values in val_losses.items():
        plt.plot(values, label=f'Val {name}', linewidth=2, linestyle='--')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training and Validation Losses', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.loss_dir}/combined/losses_ep{epoch:03d}.png", dpi=300)
    plt.close()
    
    # Create individual loss plots
    for name, values in train_losses.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values, label=f'Train {name}', linewidth=2, color='blue')
        if name in val_losses:
            plt.plot(val_losses[name], label=f'Val {name}', linewidth=2, color='red', linestyle='--')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'{name} Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{cfg.loss_dir}/individual/{name}_ep{epoch:03d}.png", dpi=300)
        plt.close()

def visualize_attention_maps(model, input_img, epoch, idx=0):
    """Visualize the attention maps from the model's self-attention layers"""
    # Store current model state
    training_state = model.training
    
    # Set model to evaluation mode temporarily
    model.eval()
    
    # Create a forward hook to extract attention maps
    attention_maps = {}
    
    def get_attention_map(name):
        def hook_fn(module, input, output):
            # Get the attention map from module.attention_map
            if hasattr(module, 'attention_map'):
                attention_maps[name] = module.attention_map
        return hook_fn
    
    # Register hooks for attention layers
    hooks = []
    hooks.append(model.attn2.register_forward_hook(get_attention_map('attn2')))
    hooks.append(model.attn3.register_forward_hook(get_attention_map('attn3')))
    hooks.append(model.attn4.register_forward_hook(get_attention_map('attn4')))
    
    # Forward pass to collect attention maps
    with torch.no_grad():
        _ = model(input_img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Sample image for visualization
    sample_img = ((input_img[0].detach().cpu() + 1) / 2).permute(1, 2, 0).numpy()
    
    # Visualize attention maps
    for name, attn_map in attention_maps.items():
        # Reshape to spatial dimensions (first sample only)
        h = w = int(np.sqrt(attn_map[0].shape[0]))
        attn_vis = attn_map[0].reshape(h, w, h, w)
        
        # Average attention across all possible points
        attn_avg = attn_vis.mean(dim=(0, 1)).cpu().numpy()
        
        # Plot attention heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(sample_img)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(attn_avg, cmap='hot')
        plt.title(f'Attention Map ({name})')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{cfg.attn_dir}/ep{epoch:03d}_{name}_sample{idx}_attn.png", dpi=300)
        plt.close()
    
    # Restore original model state
    if training_state:
        model.train()
    
    return attention_maps

def visualize_model_components(input_img, gt_img, main_output, detail_output, fusion_output, alpha_value, epoch, idx):
    """Visualize different model components and their outputs"""
    # Convert tensors to numpy arrays for visualization - add detach() to all tensors
    input_np = ((input_img.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    gt_np = ((gt_img.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    main_np = ((main_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    detail_np = ((detail_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    fusion_np = ((fusion_output.detach().cpu() + 1) / 2).permute(0, 2, 3, 1).numpy()
    
    # Calculate residual (detail enhancement)
    residual = (detail_output - main_output).detach().cpu()
    # Normalize residual for better visualization
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    residual_np = residual.permute(0, 2, 3, 1).numpy()
    
    # Choose sample to visualize (first in batch)
    sample_idx = 0
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(input_np[sample_idx])
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(main_np[sample_idx])
    plt.title('Main Generator Output')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(detail_np[sample_idx])
    plt.title('DetailNet Output')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(residual_np[sample_idx])
    plt.title('Detail Enhancement (Residual)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(fusion_np[sample_idx])
    plt.title(f'Fusion Output (α={alpha_value:.2f})')  # Fixed this line
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(gt_np[sample_idx])
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{cfg.comp_dir}/ep{epoch:03d}_b{idx:04d}_comp.png", dpi=300)
    plt.close()
    
    # Save individual outputs for all samples in batch
    batch_size = input_img.size(0)
    for i in range(min(batch_size, 4)):  # Show at most 4 samples
        grid = vutils.make_grid(
            torch.cat([
                input_img[i:i+1].detach(), 
                main_output[i:i+1].detach(), 
                detail_output[i:i+1].detach(), 
                fusion_output[i:i+1].detach(), 
                gt_img[i:i+1].detach()
            ], 0),
            nrow=5, normalize=True, scale_each=True
        )
        vutils.save_image(grid, f"{cfg.output_dir}/ep{epoch:03d}_b{idx:04d}_sample{i}.png")

# ────────────────────────────────────────────────────
# 4. Train and Validation
# ────────────────────────────────────────────────────
def train():
    # Data loaders
    tr_loader, val_loader = make_loader("train"), make_loader("val")
    
    # Initialize the network
    netG = UNetGeneratorPlus().to(cfg.device) # Main Generator part
    netD = PatchDiscriminator().to(cfg.device) # Patch Discriminator
    detailNet = DetailNet().to(cfg.device) # Residual Learning
    fusion_model = FusionModel(netG, detailNet).to(cfg.device) # Fusion Learning
    
    # Loss function
    cGAN = nn.MSELoss() # cGan loss: MSE
    cL1 = nn.L1Loss() # cL1 loss: L1
    
    # Optimizer
    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optDetail = torch.optim.Adam(detailNet.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optFusion = torch.optim.Adam([fusion_model.alpha], lr=1e-3)  
    
    # Learning rate scheduler
    lr_lambda = lambda e: 1.0 if e < 100 else 1.0 - (e-100)/(cfg.epochs-100)
    schG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda)
    schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda)
    schDetail = torch.optim.lr_scheduler.LambdaLR(optDetail, lr_lambda)

    # Initialize epoch and loss tracking variables
    start_ep = 0
    
    # Dictionaries to store losses for visualization
    train_losses = {
        'D_loss': [], 'G_loss': [], 'Detail_loss': [], 'Fusion_loss': []
    }
    val_losses = {
        'PSNR': [], 'SSIM': [], 'PSNR_Main': [], 'SSIM_Main': [], 
        'PSNR_Detail': [], 'SSIM_Detail': [], 'Alpha': []
    }
    
    # Resume training if specified
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=cfg.device)
        netG.load_state_dict(ck["G"])
        netD.load_state_dict(ck["D"])
        if "Detail" in ck:
            detailNet.load_state_dict(ck["Detail"])
        if "alpha" in ck:
            fusion_model.alpha.data = ck["alpha"]
        optG.load_state_dict(ck["optG"])
        optD.load_state_dict(ck["optD"])
        if "optDetail" in ck:
            optDetail.load_state_dict(ck["optDetail"])
        start_ep = ck["epoch"] + 1
        for _ in range(start_ep): 
            schG.step()
            schD.step() 
            schDetail.step()
        print(f"✓ Recover training from {cfg.resume} (epoch {start_ep})")

    # TensorBoard writer
    writer = SummaryWriter(cfg.log_dir)
    
    # Training loop
    for ep in range(start_ep, cfg.epochs):
        netG.train()
        netD.train()
        detailNet.train()
        
        # Track epoch losses
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        epoch_Detail_loss = 0.0
        epoch_Fusion_loss = 0.0
        num_batches = 0
        
        for i, b in enumerate(tr_loader):
            rA = b["A"].to(cfg.device) # Input Image
            rB = b["B"].to(cfg.device) # Target Image

            # -------- Training for Discriminator --------
            optD.zero_grad()
            # Fusion output for discriminator training 
            fB_main = netG(rA).detach() # main part
            fB_detail = detailNet(rA).detach() # prevent the gradient spreading to main generator
            fB_fusion = (fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail).detach() # fusion = alpha * main + (1 - alpha) * residual
            
            p_r = netD(torch.cat([rA, rB], 1)) # feature map for gt image
            p_f = netD(torch.cat([rA, fB_fusion], 1)) # feature map for generated image
            loss_D = 0.5 * (cGAN(p_r, torch.ones_like(p_r)) + cGAN(p_f, torch.zeros_like(p_f))) 
            # The first part of loss should be as close as to one as possible
            # The second part of loss should be as close to zero as possible
            # Here we use mse loss: for gt the feature map each patch value should be close to 1. 
            # For fake image the feature map each patch value should be close to 0.

            loss_D.backward()
            optD.step()

            # -------- Training for Generator --------
            optG.zero_grad()

            fB_main = netG(rA) # main part generation
            fB_detail = detailNet(rA).detach()  # prevent the gradient spreading to detailnet generator
            fB_fusion = fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail # fusion = alpha * main + (1 - alpha) * residual
             
            p_f = netD(torch.cat([rA, fB_fusion], 1)) # feature map for generated image
            loss_G = cGAN(p_f, torch.ones_like(p_f)) + cL1(fB_fusion, rB) * cfg.lambda_L1 
            # The first value should be close to 1s as close as possible
            # The second loss is L1 loss to supervise the model to generate closer to gt
            
            loss_G.backward()
            optG.step()
            
            # -------- Train the DetailNet --------
            optDetail.zero_grad()
            fB_main = netG(rA).detach()  # prevent the gradient spreading to main generator
            fB_detail = detailNet(rA)
            fB_fusion = fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail # fusion = alpha * main + (1 - alpha) * residual
            
            p_f_detail = netD(torch.cat([rA, fB_fusion], 1))
            # DetailNet loss = GAN_LOSS + L1_loss + L1(residual,detail_net) -> high frequency loss details
            loss_detail = cGAN(p_f_detail, torch.ones_like(p_f_detail)) + cL1(fB_fusion, rB) * cfg.lambda_L1 + cL1(fB_detail, rB - fB_main) * (cfg.lambda_L1 * 0.1)  # specify for residual learning
            loss_detail.backward()
            optDetail.step()
            
            # -------- Train fusion weight alpha --------
            optFusion.zero_grad()
            fB_main = netG(rA).detach() # prevent the gradient spreading to main generator
            fB_detail = detailNet(rA).detach() # prevent the gradient spreading to detail generator
            fB_fusion = fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail
            
            loss_fusion = cL1(fB_fusion, rB) * cfg.lambda_L1
            loss_fusion.backward() # specific for alpha training
            optFusion.step()
            
            # Accumulate batch losses
            epoch_D_loss += loss_D.item()
            epoch_G_loss += loss_G.item()
            epoch_Detail_loss += loss_detail.item()
            epoch_Fusion_loss += loss_fusion.item()
            num_batches += 1

            # Print progress and log to TensorBoard
            if i % 50 == 0:
                alpha_val = fusion_model.alpha.item()
                print(f"Ep {ep}/{cfg.epochs}  B {i}/{len(tr_loader)}  "
                      f"D {loss_D.item():.3f}  G {loss_G.item():.3f}  "
                      f"Detail {loss_detail.item():.3f}  Alpha {alpha_val:.3f}")
                step = ep*len(tr_loader)+i
                writer.add_scalar("Loss/D", loss_D.item(), step)
                writer.add_scalar("Loss/G", loss_G.item(), step)
                writer.add_scalar("Loss/Detail", loss_detail.item(), step)
                writer.add_scalar("Param/Alpha", alpha_val, step)
            
            # Save samples and visualizations
            if i % 100 == 0:
                # Save basic samples
                save_sample(ep, i, rA.cpu(), rB.cpu(), fB_fusion.cpu())
                
                # Save detailed component visualization
                grid_detail = vutils.make_grid(torch.cat([fB_main.cpu(), fB_detail.cpu(), fB_fusion.cpu()], 0),
                                nrow=rA.size(0), normalize=True, scale_each=True)
                vutils.save_image(grid_detail, f"{cfg.sample_dir}/ep{ep:03d}_b{i:04d}_detail.png")
                
                # Visualize model components
                fusion_psnr, fusion_ssim, main_psnr, main_ssim, detail_psnr, detail_ssim = visualize_model_components_with_metrics(
    rA, rB, fB_main, fB_detail, fB_fusion, fusion_model.alpha.item(), ep, i)
                
                # Visualize attention maps (for first image in batch)
                if i % 200 == 0:  # Less frequent since this is more computationally expensive
                    visualize_attention_maps(netG, rA[:1], ep, i)
        
        # Step learning rate schedulers
        schG.step()
        schD.step()
        schDetail.step()
        
        # Calculate average epoch losses
        avg_D_loss = epoch_D_loss / num_batches
        avg_G_loss = epoch_G_loss / num_batches
        avg_Detail_loss = epoch_Detail_loss / num_batches
        avg_Fusion_loss = epoch_Fusion_loss / num_batches
        
        # Store training losses for plotting
        train_losses['D_loss'].append(avg_D_loss)
        train_losses['G_loss'].append(avg_G_loss)
        train_losses['Detail_loss'].append(avg_Detail_loss)
        train_losses['Fusion_loss'].append(avg_Fusion_loss)
        
        # Log average epoch losses to TensorBoard
        writer.add_scalar("Epoch/D_loss", avg_D_loss, ep)
        writer.add_scalar("Epoch/G_loss", avg_G_loss, ep)
        writer.add_scalar("Epoch/Detail_loss", avg_Detail_loss, ep)
        writer.add_scalar("Epoch/Fusion_loss", avg_Fusion_loss, ep)

        # -------- Validation --------
        netG.eval()
        detailNet.eval()
        ps, ss, n = 0, 0, 0
        ps_main, ss_main = 0, 0  # Main generator metrics
        ps_detail, ss_detail = 0, 0  # DetailNet metrics
        val_sample_idx = random.randint(0, len(val_loader) - 1)  # Random batch for visualization
        
        with torch.no_grad():
            for j, b in enumerate(val_loader):
                A = b["A"].to(cfg.device)
                B = b["B"].to(cfg.device)
                
                # Calculate the metrics
                fB_main = netG(A)
                fB_detail = detailNet(A)
                fB_fusion = fusion_model.alpha * fB_main + (1 - fusion_model.alpha) * fB_detail
                
                p, s = psnr_ssim(B, fB_fusion)
                p_main, s_main = psnr_ssim(B, fB_main)
                p_detail, s_detail = psnr_ssim(B, fB_detail)
                
                bs = A.size(0)
                ps += p*bs; ss += s*bs
                ps_main += p_main*bs; ss_main += s_main*bs
                ps_detail += p_detail*bs; ss_detail += s_detail*bs
                n += bs
                
                # Save validation samples and visualizations for a random batch
                if j == val_sample_idx:
                    fusion_psnr, fusion_ssim, main_psnr, main_ssim, detail_psnr, detail_ssim = visualize_model_components_with_metrics(
    A, B, fB_main, fB_detail, fB_fusion, fusion_model.alpha.item(), ep, j+1000)
                    grid_val = vutils.make_grid(
                        torch.cat([A.cpu(), fB_main.cpu(), fB_detail.cpu(), fB_fusion.cpu(), B.cpu()], 0),
                        nrow=A.size(0), normalize=True, scale_each=True
                    )
                    vutils.save_image(grid_val, f"{cfg.comp_dir}/ep{ep:03d}_val_samples.png")
        
        # Calculate final validation metrics
        final_psnr = ps/n
        final_ssim = ss/n
        final_psnr_main = ps_main/n
        final_ssim_main = ss_main/n
        final_psnr_detail = ps_detail/n
        final_ssim_detail = ss_detail/n
        alpha_val = fusion_model.alpha.item()
        
        # Store validation metrics for plotting
        val_losses['PSNR'].append(final_psnr)
        val_losses['SSIM'].append(final_ssim)
        val_losses['PSNR_Main'].append(final_psnr_main)
        val_losses['SSIM_Main'].append(final_ssim_main)
        val_losses['PSNR_Detail'].append(final_psnr_detail)
        val_losses['SSIM_Detail'].append(final_ssim_detail)
        val_losses['Alpha'].append(alpha_val)
        
        # Print validation results
        print(f"[VAL] Fusion: PSNR {final_psnr:.2f}  SSIM {final_ssim:.3f}")
        print(f"[VAL] Main: PSNR {final_psnr_main:.2f}  SSIM {final_ssim_main:.3f}")
        print(f"[VAL] DetailNet: PSNR {final_psnr_detail:.2f}  SSIM {final_ssim_detail:.3f}")
        print(f"[VAL] Alpha {alpha_val:.3f}")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar("Val/PSNR", final_psnr, ep)
        writer.add_scalar("Val/SSIM", final_ssim, ep)
        writer.add_scalar("Val/PSNR_Main", final_psnr_main, ep)
        writer.add_scalar("Val/SSIM_Main", final_ssim_main, ep)
        writer.add_scalar("Val/PSNR_Detail", final_psnr_detail, ep)
        writer.add_scalar("Val/SSIM_Detail", final_ssim_detail, ep)
        writer.add_scalar("Val/Alpha", alpha_val, ep)
        
        # Plot and save loss curves
        plot_losses(train_losses, val_losses, ep)
        plot_metrics(val_losses, ep)
            
        # Save model checkpoints
        if (ep + 1) % 10 == 0 or ep == cfg.epochs - 1:
            visualize_validation_metrics(val_loader, netG, detailNet, fusion_model, ep)
            save_ckpt_extended(netG, netD, detailNet, fusion_model.alpha, 
                              optG, optD, optDetail, ep)
            print(f"✓ Save model's checkpoints epoch {ep}")
    
    # Close TensorBoard writer
    writer.close()

# ────────────────────────────────────────────────────
# 5. CLI
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=cfg.data_root)
    ap.add_argument("--epochs",    type=int, default=cfg.epochs)
    ap.add_argument("--resume",    default="")
    args = ap.parse_args()
    cfg.data_root = args.data_root
    cfg.epochs    = args.epochs
    cfg.resume    = args.resume
    train()