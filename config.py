from dataclasses import dataclass

@dataclass
class Config:
    # Data & Geospatial
    num_channels: int = 3               # RGB (3-band)
    scale: int = 4                      # 4x upscaling (2.0m -> 0.5m)
    reflectance_scale: float = 255.0    # 8-bit uint8 normalization
    patch_size: int = 64                # LR patch size (HR = 256x256)

    # Model (GeoAtt-RRDBNet)
    nf: int = 64
    gc: int = 32
    num_rrdb_blocks: int = 12

    # Training
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    sam_weight: float = 0.05
    device: str = "cuda"