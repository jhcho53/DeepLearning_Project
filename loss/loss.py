import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_loss(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    multi-class Dice loss
    logits: (B, C, H, W)
    masks:  (B, H, W) with integer class labels 0..C-1
    returns mean Dice loss over classes
    """
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)
    masks_onehot = F.one_hot(masks, num_classes=C)
    masks_onehot = masks_onehot.permute(0, 3, 1, 2).float()

    intersection = (probs * masks_onehot).sum(dim=(0,2,3))
    cardinality = probs.sum(dim=(0,2,3)) + masks_onehot.sum(dim=(0,2,3))
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return 1. - dice.mean()


def boundary_loss(logits: torch.Tensor, masks: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Shape-aware boundary loss over all classes using distance transform.
    For each class c, computes distance map of GT mask for class c and
    multiplies with predicted probability for class c.
    logits: (B, C, H, W)
    masks:  (B, H, W)
    """
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)
    total_loss = 0.0

    for b in range(B):
        gt = masks[b].cpu().numpy()
        sample_loss = 0.0
        for c in range(C):
            # binary mask for class c
            gt_c = (gt == c).astype(np.uint8)
            # compute distance transform: distance to class c pixels
            dist_map_c = distance_transform_edt(gt_c == 0).astype(np.float32)
            D_c = torch.from_numpy(dist_map_c).to(logits.device)  # (H, W)
            p_c = probs[b, c, :, :]  # (H, W)
            sample_loss += torch.mean(p_c * D_c)
        # average over classes
        total_loss += sample_loss / C

    return total_loss / B


class FocalLoss(nn.Module):
    """
    multi-class Focal Loss
    Args:
        alpha: Tensor[C] 클래스별 가중치 (optional)
        gamma: focusing parameter
        ignore_index: 레이블 무시 인덱스
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0,2,3,1).reshape(-1, C)
        targets_flat = targets.view(-1)
        valid = (targets_flat != self.ignore_index)
        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]

        logp = F.log_softmax(logits_flat, dim=1)
        p = torch.exp(logp)
        targets_onehot = F.one_hot(targets_flat, num_classes=C)
        logp_t = (logp * targets_onehot).sum(dim=1)
        p_t    = (p    * targets_onehot).sum(dim=1)
        alpha_t = (self.alpha.to(logits.device) * targets_onehot).sum(dim=1) if self.alpha is not None else 1.0
        loss = - alpha_t * (1 - p_t)**self.gamma * logp_t
        return loss.mean()


class WeightedFocalDiceLoss(nn.Module):
    """
    Focal + Dice + Shape-Aware Boundary Loss
    Args:
        alpha: Tensor[C] 클래스별 Focal alpha
        gamma: focal gamma
        dice_weight: float
        shape_weight: float
        ignore_index: int
    """
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        shape_weight: float = 1.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.shape_weight = shape_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, masks: torch.Tensor):
        # Focal Loss
        focal_l = self.focal(logits, masks)
        # Dice Loss
        valid_mask = (masks != self.ignore_index)
        masked_masks = masks.clone()
        masked_masks[~valid_mask] = 0
        dice_l = dice_loss(logits, masked_masks)
        # Boundary Loss
        shape_l = boundary_loss(logits, masks, ignore_index=self.ignore_index)
        # Total
        total = focal_l + self.dice_weight * dice_l + self.shape_weight * shape_l
        return total, focal_l.detach(), dice_l.detach(), shape_l.detach()
