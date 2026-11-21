import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from torch.utils.data import DataLoader
from dataset import Sentinel2GeoDataset

lr_dir = "data/dummy/train/LR"
hr_dir = "data/dummy/train/HR"
os.makedirs(lr_dir, exist_ok=True)
os.makedirs(hr_dir, exist_ok=True)

for i in range(4):
    lr_arr = np.random.randint(0, 12000, size=(4, 64, 64), dtype=np.uint16)
    hr_arr = np.random.randint(0, 12000, size=(4, 256, 256), dtype=np.uint16)
    with rasterio.open(f"{lr_dir}/tile_{i}.tif", "w", driver="GTiff", height=64, width=64,
                       count=4, dtype="uint16", crs="EPSG:32643",
                       transform=from_origin(73.85, 18.52, 10.0, 10.0)) as dst:
        dst.write(lr_arr)
    with rasterio.open(f"{hr_dir}/tile_{i}.tif", "w", driver="GTiff", height=256, width=256,
                       count=4, dtype="uint16", crs="EPSG:32643",
                       transform=from_origin(73.85, 18.52, 2.5, 2.5)) as dst:
        dst.write(hr_arr)

ds = Sentinel2GeoDataset(lr_dir, hr_dir, scale=4, patch_size=32, num_channels=4, is_train=True)
loader = DataLoader(ds, batch_size=2, shuffle=True)
lr_batch, hr_batch = next(iter(loader))

print(f"LR Batch: {lr_batch.shape} | min={lr_batch.min():.4f}, max={lr_batch.max():.4f}, dtype={lr_batch.dtype}")
print(f"HR Batch: {hr_batch.shape} | min={hr_batch.min():.4f}, max={hr_batch.max():.4f}, dtype={hr_batch.dtype}")
assert lr_batch.shape == (2, 4, 32, 32)
assert hr_batch.shape == (2, 4, 128, 128)
assert lr_batch.max() <= 1.0 and hr_batch.max() <= 1.0
print("Dataset preprocessing test PASSED!")