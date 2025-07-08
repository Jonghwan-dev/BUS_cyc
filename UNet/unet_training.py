import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

class NPZSegDataset(Dataset):
    """Dataset wrapping a pre‑processed .npz segmentation dataset."""

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.images = data["images"].astype(np.float32) / 255.0  # 0‑1 normalize
        self.masks = data["masks"].astype(np.float32)
        # Ensure masks are (N, H, W, 1)
        if self.masks.ndim == 3:
            self.masks = self.masks[..., None]
        # If mask values are not {0,1}, binarize (>0 -> 1)
        self.masks = (self.masks > 0).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(2, 0, 1)  # HWC -> CHW
        mask = self.masks[idx].transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img), torch.from_numpy(mask)

# -----------------------------------------------------------------------------
# Model: A slightly deeper UNet variant (5 encoder blocks)
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_c=64):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_c)
        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        self.enc4 = DoubleConv(base_c * 4, base_c * 8)
        self.enc5 = DoubleConv(base_c * 8, base_c * 16)
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_c * 16, base_c * 8)
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c * 2, base_c)
        self.head = nn.Conv2d(base_c, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.head(d1))

# -----------------------------------------------------------------------------
# Loss & Metrics
# -----------------------------------------------------------------------------

def focal_loss(pred, target, alpha=0.8, gamma=2.0, eps=1e-6):
    """Binary focal loss."""
    p = pred.clamp(eps, 1 - eps)
    pt = torch.where(target == 1, p, 1 - p)
    w = alpha * (1 - pt) ** gamma
    return -(w * torch.log(pt)).mean()

@torch.inference_mode()
def compute_metrics(pred, target, thr=0.5):
    pred_bin = (pred > thr).float()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item()

    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return acc, dice, iou

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, msks in tqdm(loader, desc="train", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = focal_loss(preds, msks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    loss_sum = acc_sum = dice_sum = iou_sum = 0.0
    for imgs, msks in tqdm(loader, desc="val", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        preds = model(imgs)
        loss_sum += focal_loss(preds, msks).item() * imgs.size(0)
        acc, dice, iou = compute_metrics(preds, msks)
        acc_sum += acc * imgs.size(0)
        dice_sum += dice * imgs.size(0)
        iou_sum += iou * imgs.size(0)
    n = len(loader.dataset)
    return {
        "loss": loss_sum / n,
        "acc": acc_sum / n,
        "dice": dice_sum / n,
        "iou": iou_sum / n,  # 표기 그대로 사용
    }

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = NPZSegDataset(args.data)
    val_len = int(len(ds) * 0.2)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=3, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
            f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

    # Save final model & history
    out = Path(args.data).with_suffix("")
    torch.save(model.state_dict(), f"{out}_unet.pth")
    with open(f"{out}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Model & history saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="path to .npz file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
