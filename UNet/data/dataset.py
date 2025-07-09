import torch
from torch.utils.data import Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self, processed):
        self.data = processed
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]['image']
        mask = self.data[idx]['mask']
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        if mask is not None:
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = np.zeros_like(img)
        return torch.tensor(img), torch.tensor(mask)