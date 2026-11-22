import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from config import Config
from dataset import Sentinel2GeoDataset
from model import GeoAttRRDBNet

def calc_metrics(pred, target, scale=4):
    pred = np.clip(pred, 0.0, 1.0)
    mse = np.mean((pred - target) ** 2)
    psnr = 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
    rmse = math.sqrt(mse)
    ssim_val = ssim(pred.transpose(1, 2, 0), target.transpose(1, 2, 0), data_range=1.0, channel_axis=2)

    # SAM (degrees)
    dot = np.sum(pred * target, axis=0)
    norm = np.linalg.norm(pred, axis=0) * np.linalg.norm(target, axis=0) + 1e-8
    sam = np.mean(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))

    # ERGAS
    ergas_terms = [np.mean((pred[c] - target[c]) ** 2) / (np.mean(target[c]) ** 2 + 1e-8) for c in range(pred.shape[0])]
    ergas = (100.0 / scale) * math.sqrt(np.mean(ergas_terms))
    return psnr, ssim_val, rmse, sam, ergas

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ds = Sentinel2GeoDataset(
        lr_dir=args.test_lr, hr_dir=args.test_hr,
        scale=args.scale, num_channels=args.num_channels,
        reflectance_scale=args.reflectance_scale,
        cities=args.cities, is_train=False
    )

    model = GeoAttRRDBNet(
        in_nc=args.num_channels, out_nc=args.num_channels,
        nf=args.nf, nb=args.nb, gc=args.gc, scale=args.scale
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"Evaluating on {len(ds)} test tiles (device: {device})...")
    bic_m, sr_m = [], []

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Evaluating"):
            lr, hr = ds[i]
            lr_b = lr.unsqueeze(0).to(device)
            bic = F.interpolate(lr_b, scale_factor=args.scale, mode="bicubic", align_corners=False).squeeze(0).cpu().numpy()
            sr = model(lr_b).squeeze(0).cpu().numpy()
            hr_np = hr.numpy()

            bic_m.append(calc_metrics(bic, hr_np, args.scale))
            sr_m.append(calc_metrics(sr, hr_np, args.scale))

    bm = np.mean(bic_m, axis=0)
    sm = np.mean(sr_m, axis=0)
    print(f"Bicubic Baseline   | PSNR: {bm[0]:.2f} dB | SSIM: {bm[1]:.4f} | RMSE: {bm[2]:.4f} | SAM: {bm[3]:.2f}° | ERGAS: {bm[4]:.2f}")
    print(f"GeoAtt-RRDBNet     | PSNR: {sm[0]:.2f} dB | SSIM: {sm[1]:.4f} | RMSE: {sm[2]:.4f} | SAM: {sm[3]:.2f}° | ERGAS: {sm[4]:.2f}")

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_lr", required=True)
    parser.add_argument("--test_hr", required=True)
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.num_channels)
    parser.add_argument("--reflectance_scale", type=float, default=cfg.reflectance_scale)
    parser.add_argument("--nf", type=int, default=cfg.nf)
    parser.add_argument("--gc", type=int, default=cfg.gc)
    parser.add_argument("--nb", type=int, default=cfg.num_rrdb_blocks)
    parser.add_argument("--device", default=cfg.device)
    evaluate(parser.parse_args())