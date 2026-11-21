import os
import csv
import math
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import Sentinel2GeoDataset
from model import GeoAttRRDBNet
from losses import SpectralSpatialLoss

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Training on device: {device}")

    train_ds = Sentinel2GeoDataset(
        lr_dir=args.train_lr, hr_dir=args.train_hr,
        scale=args.scale, patch_size=args.patch_size,
        num_channels=args.num_channels, reflectance_scale=args.reflectance_scale,
        cities=args.train_cities, is_train=True
    )
    val_ds = Sentinel2GeoDataset(
        lr_dir=args.val_lr, hr_dir=args.val_hr,
        scale=args.scale, patch_size=args.patch_size,
        num_channels=args.num_channels, reflectance_scale=args.reflectance_scale,
        cities=args.val_cities, is_train=False
    )
    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = GeoAttRRDBNet(
        in_nc=args.num_channels, out_nc=args.num_channels,
        nf=args.nf, nb=args.nb, gc=args.gc, scale=args.scale
    ).to(device)

    criterion = SpectralSpatialLoss(sam_weight=args.sam_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best PSNR: {best_psnr:.2f} dB)")

    log_path = os.path.join(args.out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_psnr", "lr"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr_img = model(lr_img)
                mse = torch.mean((sr_img - hr_img) ** 2).item()
                val_psnr += 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
        val_psnr /= len(val_loader)

        print(f"Epoch [{epoch}/{args.epochs}] | Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | LR: {current_lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{val_psnr:.4f}", f"{current_lr:.6e}"])

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr
        }, os.path.join(args.out_dir, "last.pth"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_lr", required=True)
    parser.add_argument("--train_hr", required=True)
    parser.add_argument("--val_lr", required=True)
    parser.add_argument("--val_hr", required=True)
    parser.add_argument("--train_cities", nargs="+", default=None)
    parser.add_argument("--val_cities", nargs="+", default=None)
    parser.add_argument("--out_dir", default="./runs/geoatt_rrdb")
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.num_channels)
    parser.add_argument("--reflectance_scale", type=float, default=cfg.reflectance_scale)
    parser.add_argument("--patch_size", type=int, default=cfg.patch_size)
    parser.add_argument("--nf", type=int, default=cfg.nf)
    parser.add_argument("--gc", type=int, default=cfg.gc)
    parser.add_argument("--nb", type=int, default=cfg.num_rrdb_blocks)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--sam_weight", type=float, default=cfg.sam_weight)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--resume", default=None)
    train(parser.parse_args())