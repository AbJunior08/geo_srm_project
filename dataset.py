import os
import glob
import random
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

def normalize_reflectance(img: np.ndarray, scale_factor: float = 255.0) -> np.ndarray:
    img = img.astype(np.float32) / scale_factor
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0)

class Sentinel2GeoDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 4,
                 patch_size: int = 64, num_channels: int = 3,
                 reflectance_scale: float = 255.0, cities=None, is_train: bool = True):
        lr_all = sorted(glob.glob(os.path.join(lr_dir, "*.tif")))
        hr_all = sorted(glob.glob(os.path.join(hr_dir, "*.tif")))

        if cities is not None:
            lr_all = [f for f in lr_all if any(os.path.basename(f).startswith(c) for c in cities)]
            hr_all = [f for f in hr_all if any(os.path.basename(f).startswith(c) for c in cities)]

        self.lr_files = lr_all
        self.hr_files = hr_all
        assert len(self.lr_files) == len(self.hr_files) and len(self.lr_files) > 0, \
            f"Mismatch or empty LR ({len(self.lr_files)}) / HR ({len(self.hr_files)}) directories."
        self.scale = scale
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.reflectance_scale = reflectance_scale
        self.is_train = is_train

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        with rasterio.open(self.lr_files[idx]) as src_lr:
            lr = src_lr.read()[:self.num_channels]
        with rasterio.open(self.hr_files[idx]) as src_hr:
            hr = src_hr.read()[:self.num_channels]

        lr = normalize_reflectance(lr, self.reflectance_scale)
        hr = normalize_reflectance(hr, self.reflectance_scale)

        _, h_lr, w_lr = lr.shape
        hr = hr[:, :h_lr * self.scale, :w_lr * self.scale]

        if self.is_train:
            ps = self.patch_size
            top = random.randint(0, h_lr - ps)
            left = random.randint(0, w_lr - ps)
            lr = lr[:, top:top + ps, left:left + ps]
            hr = hr[:, top * self.scale:(top + ps) * self.scale,
                    left * self.scale:(left + ps) * self.scale]

            if random.random() < 0.5:
                lr, hr = np.flip(lr, axis=2).copy(), np.flip(hr, axis=2).copy()
            if random.random() < 0.5:
                lr, hr = np.flip(lr, axis=1).copy(), np.flip(hr, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                lr, hr = np.rot90(lr, k, axes=(1, 2)).copy(), np.rot90(hr, k, axes=(1, 2)).copy()
        else:
            # Safe 256x256 LR (1024x1024 HR) center crop for GPU memory safety during val/eval
            val_ps = min(256, h_lr, w_lr)
            top = (h_lr - val_ps) // 2
            left = (w_lr - val_ps) // 2
            lr = lr[:, top:top + val_ps, left:left + val_ps]
            hr = hr[:, top * self.scale:(top + val_ps) * self.scale,
                    left * self.scale:(left + val_ps) * self.scale]

        return torch.from_numpy(lr), torch.from_numpy(hr)