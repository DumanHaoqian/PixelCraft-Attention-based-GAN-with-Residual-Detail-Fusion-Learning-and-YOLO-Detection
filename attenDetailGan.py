"""
===================================================================
PixelCraft (Improved U-Net + ResBlock + Self-Attention + DetailNet)
without visualization
===================================================================
"""

import argparse, random
from pathlib import Path
from typing import Union

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
    data_root   = "ds_copy" #The refernece dataset          
    ckpt_dir    = "checkpoints"
    sample_dir  = "samples"
    log_dir     = "runs/pix2pix_plus_detail"

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
    def forward(self, x):
        B,C,H,W = x.size()
        q = self.query(x).view(B, -1, H*W)          # B,Cq,N
        k = self.key(x).view(B, -1, H*W)            # B,Ck,N
        v = self.value(x).view(B, -1, H*W)          # B,C ,N
        attn = torch.bmm(q.permute(0,2,1), k)       # B,N,N
        attn = F.softmax(attn/(C**0.5), dim=-1)
        out  = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)
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

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); e8=self.e8(e7)
        b  = self.res(e8)

        d1 = self.d1(b);          d1 = torch.cat([d1, e7], 1)
        d2 = self.d2(d1);         d2 = torch.cat([d2, e6], 1)
        d3 = self.d3(d2);         d3 = torch.cat([d3, e5], 1)
        d4 = self.d4(d3);         d4 = self.attn4(torch.cat([d4, e4], 1))
        d5 = self.d5(d4);         d5 = self.attn3(torch.cat([d5, e3], 1))
        d6 = self.d6(d5);         d6 = self.attn2(torch.cat([d6, e2], 1))
        d7 = self.d7(d6);         d7 = torch.cat([d7, e1], 1)
        out= self.d8(d7)
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
        self.res_blocks = nn.Sequential(
            ResBlock(nf*4),
            ResBlock(nf*4),
            SelfAttn(nf*4)  # Self-attention mechanism helps capture global information.
        )
        
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
    
    def forward(self, x):
        feat = self.enc(x)
        feat = self.res_blocks(feat)
        out = self.dec(feat)
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
    r=((real.cpu()+1)/2).numpy().transpose(0,2,3,1)
    f=((fake.cpu()+1)/2).numpy().transpose(0,2,3,1)
    ps, ss = [], []
    for rr, ff in zip(r, f):
        ps.append(psnr(rr, ff, data_range=1))
        try: ss.append(ssim(rr, ff, channel_axis=-1, data_range=1))
        except TypeError: ss.append(ssim(rr, ff, multichannel=True, data_range=1))
    return np.mean(ps), np.mean(ss)

# ────────────────────────────────────────────────────
# 4. Train and Validation
# ────────────────────────────────────────────────────
def train():

    tr_loader, val_loader = make_loader("train"), make_loader("val")
    
    # Initialize the network
    netG = UNetGeneratorPlus().to(cfg.device) # Main Generator part
    netD = PatchDiscriminator().to(cfg.device) # Patch Discriminator
    detailNet = DetailNet().to(cfg.device) # Residual Learning
    fusionModel = FusionModel(netG, detailNet).to(cfg.device) # Fusion Learning
    
    # Loss function
    cGAN = nn.MSELoss() # cGan loss: MSE
    cL1 = nn.L1Loss() # cL1 loss: L1
    
    # Optimizer
    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optDetail = torch.optim.Adam(detailNet.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optFusion = torch.optim.Adam([fusionModel.alpha], lr=1e-3)  
    
    # Adampt the learning rate
    lr_lambda = lambda e: 1.0 if e < 100 else 1.0 - (e-100)/(cfg.epochs-100)
    schG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda)
    schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda)
    schDetail = torch.optim.lr_scheduler.LambdaLR(optDetail, lr_lambda)

    start_ep = 0
    ## Resume Traning ##
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=cfg.device)
        netG.load_state_dict(ck["G"])
        netD.load_state_dict(ck["D"])
        if "Detail" in ck:
            detailNet.load_state_dict(ck["Detail"])
        if "alpha" in ck:
            fusionModel.alpha.data = ck["alpha"]
        optG.load_state_dict(ck["optG"])
        optD.load_state_dict(ck["optD"])
        if "optDetail" in ck:
            optDetail.load_state_dict(ck["optDetail"])
        start_ep = ck["epoch"] + 1
        for _ in range(start_ep): 
            schG.step()
            schD.step() 
            schDetail.step()
        print(f"✓ Recover training from  {cfg.resume}(epoch {start_ep})")

    writer = SummaryWriter(cfg.log_dir)
    # Here training begin:
    for ep in range(start_ep, cfg.epochs):
        netG.train()
        netD.train()
        detailNet.train()
        
        for i, b in enumerate(tr_loader):
            rA = b["A"].to(cfg.device) # Input Image
            rB = b["B"].to(cfg.device) # Target Image

            # -------- Traning for Discriminator --------
            optD.zero_grad()
            # Fusion output for doscriminator training 
            fB_main = netG(rA).detach() # main part
            fB_detail = detailNet(rA).detach() # prevent the gradient spreading to main generator
            fB_fusion = (fusionModel.alpha * fB_main + (1 - fusionModel.alpha) * fB_detail).detach() # fusion = alpha * main + ( 1 - alpha) * resiudal
            
            p_r = netD(torch.cat([rA, rB], 1)) # feature map for gt image
            p_f = netD(torch.cat([rA, fB_fusion], 1)) # feature map for generated image
            loss_D = 0.5 * (cGAN(p_r, torch.ones_like(p_r)) + cGAN(p_f, torch.zeros_like(p_f))) 
            # The first part of loss should be as close as to one as possible
            # The second part of loss should be as close to zero as possible
            # Here we use mse loss: for gt the feature map each patch value should be close to 1. 
            # For fake image the feature map each patch value should be close to 0.

            loss_D.backward()
            optD.step()

            # -------- Trinaing for Generator --------
            optG.zero_grad()

            fB_main = netG(rA) # main part generation
            fB_detail = detailNet(rA).detach()  # prevent the gradient spreading to detailnet generator
            fB_fusion = fusionModel.alpha * fB_main + (1 - fusionModel.alpha) * fB_detail # fusion = alpha * main + ( 1 - alpha) * resiudal
             
            p_f = netD(torch.cat([rA, fB_fusion], 1)) # feature map for generated image
            loss_G = cGAN(p_f, torch.ones_like(p_f)) + cL1(fB_fusion, rB) * cfg.lambda_L1 # the first value should be close to 1s as close as possible
            # the second loss is l1 loss to supervise the model to generate closer to gt
            #  
            loss_G.backward()
            optG.step()
            
            # -------- Train the  DetailNet --------
            optDetail.zero_grad()
            fB_main = netG(rA).detach()  # prevent the gradient spreading to main generator
            fB_detail = detailNet(rA)
            fB_fusion = fusionModel.alpha * fB_main + (1 - fusionModel.alpha) * fB_detail # fusion = alpha * main + ( 1 - alpha) * resiudal
            
            p_f_detail = netD(torch.cat([rA, fB_fusion], 1))
            # DetailNet loss = GAN_LOSS+L1_loss+ L1(residual,detail_net)->high frequency loss details
            loss_detail = cGAN(p_f_detail, torch.ones_like(p_f_detail)) + cL1(fB_fusion, rB) * cfg.lambda_L1 + cL1(fB_detail, rB - fB_main) * (cfg.lambda_L1 * 0.1)  # specify for residual learning
            loss_detail.backward()
            optDetail.step()
            
            # -------- train fusion weight alpha --------
            optFusion.zero_grad()
            fB_main = netG(rA).detach() # prevent the gradient spreading to main generator
            fB_detail = detailNet(rA).detach() # prevent the gradient spreading to detail generator
            fB_fusion = fusionModel.alpha * fB_main + (1 - fusionModel.alpha) * fB_detail
            
            loss_fusion = cL1(fB_fusion, rB) * cfg.lambda_L1
            loss_fusion.backward() # specifical for alpha training
            optFusion.step()

            if i % 50 == 0:
                alpha_val = fusionModel.alpha.item()
                print(f"Ep {ep}/{cfg.epochs}  B {i}/{len(tr_loader)}  "
                      f"D {loss_D.item():.3f}  G {loss_G.item():.3f}  "
                      f"Detail {loss_detail.item():.3f}  Alpha {alpha_val:.3f}")
                step = ep*len(tr_loader)+i
                writer.add_scalar("Loss/D", loss_D.item(), step)
                writer.add_scalar("Loss/G", loss_G.item(), step)
                writer.add_scalar("Loss/Detail", loss_detail.item(), step)
                writer.add_scalar("Param/Alpha", alpha_val, step)
            if i % 200 == 0:
                save_sample(ep, i, rA.cpu(), rB.cpu(), fB_fusion.cpu())
                grid_detail = vutils.make_grid(torch.cat([fB_main.cpu(), fB_detail.cpu(), fB_fusion.cpu()], 0),
                                nrow=rA.size(0), normalize=True, scale_each=True)
                vutils.save_image(grid_detail, f"{cfg.sample_dir}/ep{ep:03d}_b{i:04d}_detail.png")

        schG.step()
        schD.step()
        schDetail.step()

        # -------- Validation --------
        netG.eval()
        detailNet.eval()
        ps, ss, n = 0, 0, 0
        ps_main, ss_main = 0, 0  # Main generator matrics
        ps_detail, ss_detail = 0, 0  # DetailNet Matrics
        with torch.no_grad():
            for b in val_loader:
                A = b["A"].to(cfg.device)
                B = b["B"].to(cfg.device)
                
                # Calculate the matrics
                fB_main = netG(A)
                fB_detail = detailNet(A)
                fB_fusion = fusionModel.alpha * fB_main + (1 - fusionModel.alpha) * fB_detail
                
                p, s = psnr_ssim(B, fB_fusion)
                p_main, s_main = psnr_ssim(B, fB_main)
                p_detail, s_detail = psnr_ssim(B, fB_detail)
                
                bs = A.size(0)
                ps += p*bs; ss += s*bs
                ps_main += p_main*bs; ss_main += s_main*bs
                ps_detail += p_detail*bs; ss_detail += s_detail*bs
                n += bs
                
        print(f"[VAL] Fusion: PSNR {ps/n:.2f}  SSIM {ss/n:.3f}")
        print(f"[VAL] Main: PSNR {ps_main/n:.2f}  SSIM {ss_main/n:.3f}")
        print(f"[VAL] DetailNet: PSNR {ps_detail/n:.2f}  SSIM {ss_detail/n:.3f}")
        print(f"[VAL] Alpha {fusionModel.alpha.item():.3f}")
        
        writer.add_scalar("Val/PSNR", ps/n, ep)
        writer.add_scalar("Val/SSIM", ss/n, ep)
        writer.add_scalar("Val/PSNR_Main", ps_main/n, ep)
        writer.add_scalar("Val/SSIM_Main", ss_main/n, ep)
        writer.add_scalar("Val/PSNR_Detail", ps_detail/n, ep)
        writer.add_scalar("Val/SSIM_Detail", ss_detail/n, ep)
        writer.add_scalar("Val/Alpha", fusionModel.alpha.item(), ep)

        if (ep + 1) % 10 == 0 or ep == cfg.epochs - 1:
            save_ckpt_extended(netG, netD, detailNet, fusionModel.alpha, 
                             optG, optD, optDetail, ep)
            print(f"✓ Save model's checkpoints epoch {ep}")
    
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