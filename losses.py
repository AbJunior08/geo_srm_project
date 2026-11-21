import torch
import torch.nn as nn

class SpectralSpatialLoss(nn.Module):
    def __init__(self, sam_weight: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sam_weight = sam_weight
        self.eps = eps

    def sam_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(sr * hr, dim=1)
        norm_sr = torch.linalg.norm(sr, dim=1) + self.eps
        norm_hr = torch.linalg.norm(hr, dim=1) + self.eps
        # Clamp strictly inside (-1, 1) so acos gradients never become NaN
        cos_theta = torch.clamp(dot / (norm_sr * norm_hr), -1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos_theta)
        return torch.mean(angle)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor):
        loss_l1 = self.l1(sr, hr)
        loss_sam = self.sam_loss(sr, hr)
        return loss_l1 + self.sam_weight * loss_sam