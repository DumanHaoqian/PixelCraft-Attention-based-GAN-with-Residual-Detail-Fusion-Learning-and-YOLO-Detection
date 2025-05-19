import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from typing import Union

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.tensorboard import SummaryWriter

class Config:
    data_root = "/home/haoqian/gan/lc/ds_copy"
    ckpt_dir  = "checkpoints"
    sample_dir= "samples"
    log_dir   = "runs/pix2pix"

    epochs      = 200
    batch_size  = 8
    num_workers = 4
    lr          = 2e-4
    beta1       = 0.5
    lambda_L1   = 100
    img_size    = 256
    resume      = ""         
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()


Path(cfg.ckpt_dir).mkdir(exist_ok=True, parents=True)
Path(cfg.sample_dir).mkdir(exist_ok=True, parents=True)


class SemanticRealPairDataset(Dataset):
    def __init__(self,root: Union[str, Path], phase: str = "train", img_size: int = 256):
        super().__init__()

        root = Path(root)
        if phase == "train":
            self.rootA = root / "input_train"
            self.rootB = root / "target_train"
        elif phase == "val":
            self.rootA = root / "input_val"
            self.rootB = root / "target_val"
        else:
            raise ValueError("phase can only be 'train' or 'val' or ;test" )

        self.paths = sorted(list(self.rootA.glob("*")))
        assert len(self.paths) != 0, f"can't find {self.rootA}"

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5 if phase == "train" else 0.0),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # [-1,1]
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        pathA = self.paths[idx]
        pathB = self.rootB / pathA.name
        imgA  = Image.open(pathA).convert("RGB")
        imgB  = Image.open(pathB).convert("RGB")
        return {
            "A": self.transform(imgA),
            "B": self.transform(imgB),
            "name": pathA.stem
        }

def make_dataloader(phase: str):
    ds = SemanticRealPairDataset(cfg.data_root, phase, cfg.img_size)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=(phase == "train"),
                      num_workers=cfg.num_workers,
                      pin_memory=True)


def init_weights(m, gain=0.02):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, gain)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act="relu", dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = UNetBlock(ngf, ngf * 2, down=True, act="lrelu")
        self.e3 = UNetBlock(ngf * 2, ngf * 4, down=True, act="lrelu")
        self.e4 = UNetBlock(ngf * 4, ngf * 8, down=True, act="lrelu")
        self.e5 = UNetBlock(ngf * 8, ngf * 8, down=True, act="lrelu")
        self.e6 = UNetBlock(ngf * 8, ngf * 8, down=True, act="lrelu")
        self.e7 = UNetBlock(ngf * 8, ngf * 8, down=True, act="lrelu")
        self.e8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.ReLU(True))
        # Decoder
        self.d1 = UNetBlock(ngf * 8, ngf * 8, down=False, dropout=True)
        self.d2 = UNetBlock(ngf * 16, ngf * 8, down=False, dropout=True)
        self.d3 = UNetBlock(ngf * 16, ngf * 8, down=False, dropout=True)
        self.d4 = UNetBlock(ngf * 16, ngf * 8, down=False)
        self.d5 = UNetBlock(ngf * 16, ngf * 4, down=False)
        self.d6 = UNetBlock(ngf * 8, ngf * 2, down=False)
        self.d7 = UNetBlock(ngf * 4, ngf, down=False)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1),
            nn.Tanh())
        self.apply(init_weights)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6); bottleneck = self.e8(e7)
        d1 = self.d1(bottleneck); d1 = torch.cat([d1, e7], 1)
        d2 = self.d2(d1);         d2 = torch.cat([d2, e6], 1)
        d3 = self.d3(d2);         d3 = torch.cat([d3, e5], 1)
        d4 = self.d4(d3);         d4 = torch.cat([d4, e4], 1)
        d5 = self.d5(d4);         d5 = torch.cat([d5, e3], 1)
        d6 = self.d6(d5);         d6 = torch.cat([d6, e2], 1)
        d7 = self.d7(d6);         d7 = torch.cat([d7, e1], 1)
        return self.d8(d7)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),       # 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),              # 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),              # 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),              # 31
            nn.Conv2d(ndf * 8, 1, 4, 1, 1)                                 # 30 → Patch
        )
        self.apply(init_weights)

    def forward(self, x): return self.net(x)


def save_sample(epoch, batch_idx, realA, realB, fakeB):
    grid = vutils.make_grid(torch.cat([realA, fakeB, realB], 0),
                            nrow=realA.size(0), normalize=True, scale_each=True)
    vutils.save_image(grid, f"{cfg.sample_dir}/epoch_{epoch:03d}_batch_{batch_idx:04d}.png")

def save_checkpoint(netG, netD, optG, optD, epoch):
    torch.save({
        "G": netG.state_dict(),
        "D": netD.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "epoch": epoch
    }, f"{cfg.ckpt_dir}/pix2pix_epoch_{epoch:03d}.pth")

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

def train():

    train_loader = make_dataloader("train")
    val_loader   = make_dataloader("val")


    netG = UNetGenerator().to(cfg.device)
    netD = PatchDiscriminator().to(cfg.device)

    criterion_GAN = nn.BCEWithLogitsLoss()  
    criterion_MSE = nn.MSELoss()            


    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))


    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=cfg.device)
        netG.load_state_dict(ckpt["G"]); netD.load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"]); optD.load_state_dict(ckpt["optD"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {cfg.resume}, epoch {start_epoch}")

    writer = SummaryWriter(cfg.log_dir)

    for epoch in range(start_epoch, cfg.epochs):
        netG.train(); netD.train()
        for i, data in enumerate(train_loader):
            real_A = data["A"].to(cfg.device)
            real_B = data["B"].to(cfg.device)

            # ====== Train D ======
            optD.zero_grad()
            with torch.no_grad():
                fake_B_detach = netG(real_A).detach()
            pred_real = netD(torch.cat([real_A, real_B], 1))
            pred_fake = netD(torch.cat([real_A, fake_B_detach], 1))
            loss_D = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                            criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
            loss_D.backward(); optD.step()

            # ====== Train G ======
            optG.zero_grad()
            fake_B = netG(real_A)
            pred_fake = netD(torch.cat([real_A, fake_B], 1))
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_MSE = criterion_MSE(fake_B, real_B) * cfg.lambda_L1  # 改为 MSE Loss
            loss_G = loss_G_GAN + loss_G_MSE
            loss_G.backward(); optG.step()

            if i % 50 == 0:
                print(f"Epoch {epoch}/{cfg.epochs} | Batch {i}/{len(train_loader)} "
                      f"| Loss_D {loss_D.item():.3f} | Loss_G {loss_G.item():.3f}")
                global_step = epoch * len(train_loader) + i
                writer.add_scalar("Loss/D", loss_D.item(), global_step)
                writer.add_scalar("Loss/G", loss_G.item(), global_step)

            if i % 200 == 0:
                save_sample(epoch, i, real_A.cpu(), real_B.cpu(), fake_B.cpu())

        psnr_val, ssim_val = evaluate(netG, val_loader)
        writer.add_scalar("Val/PSNR", psnr_val, epoch)
        writer.add_scalar("Val/SSIM", ssim_val, epoch)
        
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:   
            save_checkpoint(netG, netD, optG, optD, epoch)

    writer.close()


@torch.no_grad()
def evaluate(netG, val_loader):
    netG.eval()
    psnr_total, ssim_total, n = 0, 0, 0
    for data in val_loader:
        real_A = data["A"].to(cfg.device)
        real_B = data["B"].to(cfg.device)
        fake_B = netG(real_A)
        p, s = compute_metrics(real_B, fake_B)
        bs = real_A.size(0)
        psnr_total += p * bs; ssim_total += s * bs; n += bs
    psnr_avg, ssim_avg = psnr_total / n, ssim_total / n
    print(f"[VAL] PSNR={psnr_avg:.2f}  SSIM={ssim_avg:.3f}")
    return psnr_avg, ssim_avg

# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=cfg.data_root,
                        help="")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="")
    parser.add_argument("--resume", type=str, default="", help=" ckpt ")
    args = parser.parse_args()

    cfg.data_root = args.data_root
    cfg.epochs    = args.epochs
    cfg.resume    = args.resume

    train()