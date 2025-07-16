# utils/losses.py (수정된 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

class FocalLoss(nn.Module):
    # (기존과 동일)
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE
        return F_loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    # (기존과 동일)
    def __init__(self, pos_weight=1.0):
        super().__init__()
        # device 인자를 받지 않도록 수정, main에서 처리
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        # loss를 device로 이동
        self.loss.pos_weight = self.loss.pos_weight.to(inputs.device)
        return self.loss(inputs, targets)

class EdgeLoss(nn.Module):
    # (기존과 동일)
    def __init__(self):
        super().__init__()
        self.sobel = kornia.filters.Sobel()

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        pred_edge = self.sobel(inputs_sigmoid)
        target_edge = self.sobel(targets)
        return F.l1_loss(pred_edge, target_edge)

def compute_sdf(mask):
    # (기존과 동일)
    mask = mask.cpu().numpy()
    sdf = np.zeros_like(mask, dtype=np.float32)
    for b in range(mask.shape[0]):
        posmask = mask[b,0].astype(bool)
        if posmask.any(): # 마스크가 비어있지 않을 때만 계산
            negmask = ~posmask
            sdf[b,0] = edt(negmask) - edt(posmask)
    return torch.from_numpy(sdf)

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, gt):
        # --- 핵심 수정 ---
        # 이진화 대신 sigmoid 확률 값을 사용하여 미분 가능하도록 변경
        pred_sigmoid = torch.sigmoid(pred_logits) 
        
        sdf_gt = compute_sdf(gt).to(gt.device)
        # multi-class SDF loss (pred * sdf) or (pred - gt) * sdf
        # 여기서는 (pred-gt) * sdf 방식을 사용
        # abs를 두 번 사용하는 대신 (pred-gt)에 곱하는 형태로 변경
        loss = torch.mean((pred_sigmoid - gt) * sdf_gt)
        return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, bce_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_weight = bce_weight
        # pos_weight를 tensor로 미리 만들지 않고 값만 저장해두거나, 그대로 사용해도 무방합니다.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets, smooth=1e-7):
        
        # --- 핵심 수정: pos_weight를 입력 텐서와 동일한 디바이스로 이동 ---
        self.bce.pos_weight = self.bce.pos_weight.to(inputs.device)

        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        inputs_sig = torch.sigmoid(inputs)
        # Flatten a N-D Tensor to 1-D Tensor
        intersection = (inputs_sig.flatten() * targets.flatten()).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.flatten().sum() + targets.flatten().sum() + smooth)
        
        return self.bce_weight * bce_loss + self.weight * dice_loss
    
class ComboLossHD(nn.Module):
    # (기존과 동일, 내부적으로 수정된 BoundaryLoss 사용)
    def __init__(self, alpha=0.8, gamma=2, edge_weight=1.0, boundary_weight=1.0, ce_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.edge = EdgeLoss()
        self.boundary = BoundaryLoss()
        self.ce = WeightedCrossEntropyLoss(pos_weight=pos_weight)
        self.edge_weight = edge_weight
        self.boundary_weight = boundary_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        edge_loss = self.edge(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        ce_loss = self.ce(inputs, targets)

        # 각 손실의 가중치를 곱하여 최종 손실 계산
        total_loss = (
            focal_loss
            + self.edge_weight * edge_loss
            + self.boundary_weight * boundary_loss
            + self.ce_weight * ce_loss
        )
        return total_loss