"""Offline evaluation metrics utilities.

This module provides functions for computing common image similarity metrics.
Metrics are implemented as independent functions in order to be easily reused.
"""

from typing import List, Dict

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def _load_image(path: str, device: torch.device, normalize: bool = True) -> torch.Tensor:
    """Load an image and convert it to a tensor.
    
    Args:
        path: Path to the image
        device: Device to load the tensor to
        normalize: If True, normalize to [0, 1] range. If False, keep as uint8.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if normalize:
        tensor = tensor.float() / 255.0
    return tensor.to(device)


def compute_speckle_snr(img: torch.Tensor) -> float:
    """Compute speckle SNR of ``img`` (higher is better)."""

    arr = img.cpu().numpy().ravel()
    return float(arr.mean() / (arr.std() + 1e-8))


def compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
    """Return the PSNR between two images."""

    return float(peak_signal_noise_ratio(real, fake, data_range=1.0))


def compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    """Return the SSIM between two images."""

    return float(structural_similarity(real, fake, channel_axis=2, data_range=1.0))


@torch.no_grad()
def compute_lpips(real: torch.Tensor, fake: torch.Tensor, lpips_fn: LearnedPerceptualImagePatchSimilarity) -> float:
    """Return LPIPS distance using the given LPIPS module."""

    return float(lpips_fn(fake.unsqueeze(0), real.unsqueeze(0)))

def _to_gray(img: torch.Tensor) -> torch.Tensor:
    """RGB(3,C,H,W) → Gray(C=1,H,W)."""
    if img.size(0) == 3:
        r, g, b = img[0], img[1], img[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(0)
    return img  # 이미 1 채널

@torch.no_grad()
def compute_lncc(
    real: torch.Tensor,
    fake: torch.Tensor,
    win: int = 9,
) -> float:
    """Local Normalized Cross-Correlation (창 크기: win×win)."""
    pad = win // 2
    N   = win * win
    real, fake = _to_gray(real).unsqueeze(0), _to_gray(fake).unsqueeze(0)  # (1,1,H,W)
    kernel = real.new_ones(1, 1, win, win)

    sum_r  = F.conv2d(real, kernel, padding=pad)
    sum_f  = F.conv2d(fake, kernel, padding=pad)
    sum_r2 = F.conv2d(real * real, kernel, padding=pad)
    sum_f2 = F.conv2d(fake * fake, kernel, padding=pad)
    sum_rf = F.conv2d(real * fake, kernel, padding=pad)

    mu_r, mu_f = sum_r / N, sum_f / N
    sig_r2 = sum_r2 / N - mu_r ** 2
    sig_f2 = sum_f2 / N - mu_f ** 2
    sig_rf = sum_rf / N - mu_r * mu_f

    lncc_map = sig_rf / torch.sqrt(sig_r2 * sig_f2 + 1e-8)
    return lncc_map.mean().item()


@torch.no_grad()
def compute_fid(real_paths: List[str], fake_paths: List[str], device: torch.device) -> float:
    """Compute FID over two image sets."""

    fid = FrechetInceptionDistance(normalize=True).to(device)
    for path in real_paths:
        fid.update(_load_image(path, device).unsqueeze(0), real=True)
    for path in fake_paths:
        fid.update(_load_image(path, device).unsqueeze(0), real=False)
    return float(fid.compute())


@torch.no_grad()
def compute_kid(real_paths: List[str], fake_paths: List[str], device: torch.device) -> float:
    """Compute KID over two image sets."""

    # Calculate subset_size as min(50, len(real_paths))
    subset_size = min(50, len(real_paths))
    kid = KernelInceptionDistance(subset_size=subset_size).to(device)
    for path in real_paths:
        kid.update(_load_image(path, device, normalize=False).unsqueeze(0), real=True)
    for path in fake_paths:
        kid.update(_load_image(path, device, normalize=False).unsqueeze(0), real=False)
    return float(kid.compute()[0])


@torch.no_grad()
def evaluate_pairwise(real_paths: List[str], fake_paths: List[str], device: torch.device) -> Dict[str, float]:
    """Evaluate standard metrics for two image lists."""

    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    psnr, ssim, lpips_score, snr = 0.0, 0.0, 0.0, 0.0
    lncc = 0.0
    n = len(real_paths)
    for r_path, f_path in zip(real_paths, fake_paths):
        real_t = _load_image(r_path, device)
        fake_t = _load_image(f_path, device)
        real_np = real_t.cpu().numpy().transpose(1, 2, 0)
        fake_np = fake_t.cpu().numpy().transpose(1, 2, 0)
        lncc += compute_lncc(real_t, fake_t)
        psnr += compute_psnr(real_np, fake_np)
        ssim += compute_ssim(real_np, fake_np)
        lpips_score += compute_lpips(real_t, fake_t, lpips_fn)
        snr += compute_speckle_snr(fake_t)

    fid_score = compute_fid(real_paths, fake_paths, device)
    kid_score = compute_kid(real_paths, fake_paths, device)

    return {
        "PSNR": psnr / n,
        "SSIM": ssim / n,
        "LPIPS": lpips_score / n,
        "Speckle_SNR": snr / n,
        "FID": fid_score,
        "KID": kid_score,
        "LNCC": lncc / n,
    }
