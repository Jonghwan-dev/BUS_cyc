import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torch.utils.data import DataLoader
from UNet.models.model import UNet
from UNet.data.dataset import SegDataset
import wandb
import numpy as np

def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE
        return F_loss.mean()

class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filter"""
    def __init__(self):
        super().__init__()
        self.sobel = kornia.filters.Sobel()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        pred_edge = self.sobel(inputs)
        target_edge = self.sobel(targets)
        return F.l1_loss(pred_edge, target_edge)

class Custom_Loss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, edge_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.edge = EdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.edge_weight * self.edge(inputs, targets)

def train_unet(
    processed,
    val_ratio=0.2,
    batch_size=64,
    epochs=100,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project_name='BUS_UNet',
):
    wandb.init(project=project_name, name="unet_train_custom_loss", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_ratio": val_ratio,
    })

    n = len(processed)
    idxs = torch.randperm(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_set = SegDataset([processed[i] for i in train_idx])
    val_set = SegDataset([processed[i] for i in val_idx])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = Custom_Loss(alpha=0.8, gamma=2, edge_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        wandb.log({"train/loss": train_loss}, step=epoch)

        # Validation
        model.eval()
        val_loss = 0
        dices, accs, ious = [], [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(outputs)
                dices.append(dice_score(probs, masks))
                accs.append(pixel_accuracy(probs, masks))
                ious.append(iou_score(probs, masks))
        val_loss /= len(val_loader.dataset)
        mean_dice = np.mean(dices)
        mean_acc = np.mean(accs)
        mean_iou = np.mean(ious)

        wandb.log({
            "val/loss": val_loss,
            "val/dice": mean_dice,
            "val/pixel_acc": mean_acc,
            "val/mIoU@0.5": mean_iou,
        }, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {mean_dice:.4f} | Acc: {mean_acc:.4f} | IoU: {mean_iou:.4f}")

        # 예시 이미지 wandb에 기록 (epoch마다 1개)
        if epoch % 5 == 0 or epoch == epochs-1:
            imgs, masks = next(iter(val_loader))
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
            img_np = imgs[0,0].cpu().numpy()
            mask_np = masks[0,0].cpu().numpy()
            pred_np = (probs[0,0].cpu().numpy() > 0.5).astype(np.uint8)
            wandb.log({
                "val/example": [
                    wandb.Image(img_np, caption="Input"),
                    wandb.Image(mask_np, caption="GT"),
                    wandb.Image(pred_np, caption="Pred"),
                ]
            }, step=epoch)

    torch.save(model.state_dict(), "unet_best.pth")
    print("모델 저장 완료: unet_best.pth")
    wandb.finish()