import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model import GeoAttRRDBNet
from losses import SpectralSpatialLoss

# 1. Check full 12-block parameter count (for your conference paper table)
full_model = GeoAttRRDBNet(in_nc=4, out_nc=4, nf=64, nb=12, gc=32, scale=4)
total_params = sum(p.numel() for p in full_model.parameters()) / 1e6
print(f"Full GeoAtt-RRDBNet (12 blocks) Parameters: {total_params:.2f}M")

# 2. Fast CPU forward + backward test using 2 blocks
for scale in (2, 4):
    model = GeoAttRRDBNet(in_nc=4, out_nc=4, nf=64, nb=2, gc=32, scale=scale)
    criterion = SpectralSpatialLoss(sam_weight=0.05)

    lr = torch.rand(2, 4, 16, 16)
    hr = torch.rand(2, 4, 16 * scale, 16 * scale)

    sr = model(lr)
    loss = criterion(sr, hr)
    loss.backward()

    assert sr.shape == hr.shape, f"Shape mismatch: {sr.shape} vs {hr.shape}"
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
    grad_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grad_ok, "Found NaN or Inf in gradients!"

    print(f"Scale {scale}x OK | SR Shape: {tuple(sr.shape)} | Loss: {loss.item():.4f} | Gradients finite: {grad_ok}")

print("Model & Loss test PASSED!")