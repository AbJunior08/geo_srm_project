import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from config import Config
from dataset import Sentinel2GeoDataset
from model import GeoAttRRDBNet

def compute_metrics(sr: np.ndarray, hr: np.ndarray, scale: int = 4):
    """sr, hr: numpy arrays of shape (C, H, W) in [0, 1]."""
    eps = 1e-8
    mse = np.mean((sr - hr) ** 2)
    rmse = math.sqrt(mse)
    psnr = 100.0 if mse == 0 else 20 * math.log10(1.0 / rmse)

    # SSIM (channel-last for skimage)
    ssim_val = ssim(sr.transpose(1, 2, 0), hr.transpose(1, 2, 0), data_range=1.0, channel_axis=2)

    # SAM (Spectral Angle Mapper in degrees)
    dot = np.sum(sr * hr, axis=0)
    norm_sr = np.linalg.norm(sr, axis=0) + eps
    norm_hr = np.linalg.norm(hr, axis=0) + eps
    sam_deg = np.mean(np.degrees(np.arccos(np.clip(dot / (norm_sr * norm_hr), -1.0, 1.0))))

    # ERGAS
    band_errors = []
    for c in range(sr.shape[0]):
        b_rmse = math.sqrt(np.mean((sr[c] - hr[c]) ** 2))
        b_mean = np.mean(hr[c]) + eps
        band_errors.append((b_rmse / b_mean) ** 2)
    ergas = (100.0 / scale) * math.sqrt(np.mean(band_errors))

    # NDVI MAE (only computed for 4+ band multispectral imagery with Red=2, NIR=3)
    ndvi_mae = None
    if sr.shape[0] >= 4:
        ndvi_sr = (sr[3] - sr[2]) / (sr[3] + sr[2] + eps)
        ndvi_hr = (hr[3] - hr[2]) / (hr[3] + hr[2] + eps)
        ndvi_mae = np.mean(np.abs(ndvi_sr - ndvi_hr))

    return {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "RMSE": rmse,
        "SAM": sam_deg,
        "ERGAS": ergas,
        "NDVI_MAE": ndvi_mae,
    }

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ds = Sentinel2GeoDataset(
        lr_dir=args.test_lr,
        hr_dir=args.test_hr,
        scale=args.scale,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        reflectance_scale=args.reflectance_scale,
        cities=args.cities,
        is_train=False,
    )
    print(f"Evaluating on {len(ds)} test tiles (device: {device})...")

    model = GeoAttRRDBNet(
        in_nc=args.num_channels,
        out_nc=args.num_channels,
        nf=args.nf,
        nb=args.nb,
        gc=args.gc,
        scale=args.scale,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    model_scores, bicubic_scores = [], []
    with torch.no_grad():
        for lr, hr in ds:
            lr_b = lr.unsqueeze(0).to(device)
            sr = model(lr_b).squeeze(0).cpu().numpy()
            bic = (
                F.interpolate(lr_b, scale_factor=args.scale, mode="bicubic", align_corners=False)
                .clamp(0, 1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            hr_np = hr.numpy()

            model_scores.append(compute_metrics(sr, hr_np, args.scale))
            bicubic_scores.append(compute_metrics(bic, hr_np, args.scale))

    for name, scores in [("Bicubic Baseline", bicubic_scores), ("GeoAtt-RRDBNet", model_scores)]:
        avg = {k: np.mean([s[k] for s in scores]) for k in scores[0] if scores[0][k] is not None}
        ndvi_str = f" | NDVI MAE: {avg['NDVI_MAE']:.4f}" if "NDVI_MAE" in avg else ""
        print(
            f"{name:18s} | PSNR: {avg['PSNR']:.2f} dB | SSIM: {avg['SSIM']:.4f} | "
            f"RMSE: {avg['RMSE']:.4f} | SAM: {avg['SAM']:.2f}° | ERGAS: {avg['ERGAS']:.2f}{ndvi_str}"
        )

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_lr", required=True)
    parser.add_argument("--test_hr", required=True)
    parser.add_argument("--cities", nargs="+", default=None, help="Optional city prefixes to filter test split (e.g., --cities vienna)")
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.num_channels)
    parser.add_argument("--reflectance_scale", type=float, default=cfg.reflectance_scale)
    parser.add_argument("--patch_size", type=int, default=cfg.patch_size)
    parser.add_argument("--nf", type=int, default=cfg.nf)
    parser.add_argument("--gc", type=int, default=cfg.gc)
    parser.add_argument("--nb", type=int, default=cfg.num_rrdb_blocks)
    parser.add_argument("--device", default=cfg.device)
    evaluate(parser.parse_args())