import os
import argparse
import numpy as np
import rasterio
import torch

from config import Config
from dataset import normalize_reflectance
from model import GeoAttRRDBNet

def predict_with_tta(model, tile_tensor):
    preds = []
    for k in range(4):
        rot_in = torch.rot90(tile_tensor, k, dims=[2, 3])
        with torch.no_grad():
            rot_out = model(rot_in)
        preds.append(torch.rot90(rot_out, -k, dims=[2, 3]))
    stack = torch.stack(preds, dim=0)
    mean_sr = torch.mean(stack, dim=0).squeeze(0).cpu().numpy()
    uncertainty = torch.std(stack, dim=0).mean(dim=1).squeeze(0).cpu().numpy()
    return mean_sr, uncertainty

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = GeoAttRRDBNet(args.num_channels, args.num_channels, args.nf, args.nb, args.gc, args.scale).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with rasterio.open(args.input_tif) as src:
        img = normalize_reflectance(src.read()[:args.num_channels], args.reflectance_scale)
        profile = src.profile.copy()
        orig_transform = src.transform

    C, H, W = img.shape
    s = args.scale
    sr_canvas = np.zeros((C, H * s, W * s), dtype=np.float32)
    unc_canvas = np.zeros((H * s, W * s), dtype=np.float32)

    for y in range(0, H, args.tile_size):
        for x in range(0, W, args.tile_size):
            y2, x2 = min(y + args.tile_size, H), min(x + args.tile_size, W)
            tile = torch.from_numpy(img[:, y:y2, x:x2]).unsqueeze(0).to(device)
            sr_tile, unc_tile = predict_with_tta(model, tile)
            sr_canvas[:, y*s:y2*s, x*s:x2*s] = sr_tile
            unc_canvas[y*s:y2*s, x*s:x2*s] = unc_tile

    os.makedirs(os.path.dirname(args.output_sr) or ".", exist_ok=True)
    new_transform = orig_transform * orig_transform.scale(1.0 / s, 1.0 / s)
    profile.update(height=H * s, width=W * s, transform=new_transform, dtype="float32")

    profile.update(count=C)
    with rasterio.open(args.output_sr, "w", **profile) as dst:
        dst.write(sr_canvas * args.reflectance_scale)

    profile.update(count=1)
    with rasterio.open(args.output_unc, "w", **profile) as dst:
        dst.write(unc_canvas[np.newaxis, :, :])

    with rasterio.open(args.output_sr) as check:
        print(f"Saved SR GeoTIFF:  {args.output_sr} | Shape: ({check.count}, {check.height}, {check.width}) | CRS: {check.crs} | Res: {check.res}")
    print(f"Saved Uncertainty: {args.output_unc} | Mean Unc: {unc_canvas.mean():.5f}")

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_tif", required=True)
    parser.add_argument("--output_sr", required=True)
    parser.add_argument("--output_unc", required=True)
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.num_channels)
    parser.add_argument("--reflectance_scale", type=float, default=cfg.reflectance_scale)
    parser.add_argument("--nf", type=int, default=cfg.nf)
    parser.add_argument("--gc", type=int, default=cfg.gc)
    parser.add_argument("--nb", type=int, default=cfg.num_rrdb_blocks)
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument("--device", default=cfg.device)
    run_inference(parser.parse_args())