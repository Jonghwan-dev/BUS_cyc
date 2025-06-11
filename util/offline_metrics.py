"""Offline evaluation metrics utilities.

This module provides functions for computing common image similarity metrics.
Metrics are implemented as independent functions in order to be easily reused.
"""

from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def _load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load an image and convert it to a normalised tensor."""

    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
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


def compute_lpips(real: torch.Tensor, fake: torch.Tensor, lpips_fn: LearnedPerceptualImagePatchSimilarity) -> float:
    """Return LPIPS distance using the given LPIPS module."""

    return float(lpips_fn(fake.unsqueeze(0), real.unsqueeze(0)))


def compute_fid(real_paths: List[str], fake_paths: List[str], device: torch.device) -> float:
    """Compute FID over two image sets."""

    fid = FrechetInceptionDistance(normalize=True).to(device)
    for path in real_paths:
        fid.update(_load_image(path, device).unsqueeze(0), real=True)
    for path in fake_paths:
        fid.update(_load_image(path, device).unsqueeze(0), real=False)
    return float(fid.compute())


def compute_kid(real_paths: List[str], fake_paths: List[str], device: torch.device) -> float:
    """Compute KID over two image sets."""

    kid = KernelInceptionDistance(subset_size=50).to(device)
    for path in real_paths:
        kid.update(_load_image(path, device).unsqueeze(0), real=True)
    for path in fake_paths:
        kid.update(_load_image(path, device).unsqueeze(0), real=False)
    return float(kid.compute()[0])


def evaluate_pairwise(real_paths: List[str], fake_paths: List[str], device: torch.device) -> Dict[str, float]:
    """Evaluate standard metrics for two image lists."""

    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    psnr, ssim, lpips_score, snr = 0.0, 0.0, 0.0, 0.0
    n = len(real_paths)
    for r_path, f_path in zip(real_paths, fake_paths):
        real_t = _load_image(r_path, device)
        fake_t = _load_image(f_path, device)
        real_np = real_t.cpu().numpy().transpose(1, 2, 0)
        fake_np = fake_t.cpu().numpy().transpose(1, 2, 0)
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
    }

