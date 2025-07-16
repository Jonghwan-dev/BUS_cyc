# data_loader/data_loaders.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from base.base_data_loader import BaseDataLoader

# 'interpolation' 인자를 추가하여 이미지와 마스크에 다른 보간법을 적용할 수 있도록 수정
def resize_and_pad(img, target_size=(256, 256), interpolation=cv2.INTER_LINEAR):
    """
    이미지 비율을 유지하면서 리사이즈하고, 남는 공간을 패딩으로 채웁니다.
    """
    h, w = img.shape[:2]
    th, tw = target_size

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # 캔버스 생성 및 중앙 배치
    if len(img.shape) == 3:
        c = img.shape[2]
        padded_img = np.full((th, tw, c), 0, dtype=np.uint8)
    else: # Grayscale
        padded_img = np.full((th, tw), 0, dtype=np.uint8)

    start_x = (tw - new_w) // 2
    start_y = (th - new_h) // 2
    padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return padded_img

class BUSImageDataset(Dataset):
    """BUS(초음파) 이미지 데이터셋 (마스크 리사이즈 버그 수정)"""
    def __init__(self, df, is_upscale=False, transform=None):
        self.df = df
        self.is_upscale = is_upscale
        self.transform = transform
        self.target_size = (512, 512) if is_upscale else (256, 256)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row.get('mask_path')
        
        image = cv2.imread(image_path, 0) # Grayscale
        mask = cv2.imread(mask_path, 0) if pd.notnull(mask_path) else None
        
        # --- 핵심 수정: 이미지와 마스크에 다른 보간법 적용 ---
        # 이미지는 다운샘플 시 INTER_AREA, 업샘플 시 INTER_CUBIC 사용
        img_scale = min(self.target_size[1] / image.shape[1], self.target_size[0] / image.shape[0])
        img_interp = cv2.INTER_CUBIC if img_scale > 1 else cv2.INTER_AREA
        image = resize_and_pad(image, self.target_size, interpolation=img_interp)

        if mask is not None:
            # 마스크는 반드시 INTER_NEAREST 사용
            mask = resize_and_pad(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) # (1, H, W)

        if mask is not None:
            mask = (mask > 127).astype(np.float32) # 이진화
            mask = np.expand_dims(mask, axis=0)
        
        # 라벨을 정수로 매핑 (예: benign=0, malignant=1, normal=2)
        label_map = {'benign': 0, 'malignant': 1, 'normal': 2}
        label_int = label_map.get(row['label'], -1) # 라벨이 없을 경우 -1

        return {
            "image": torch.tensor(image),
            "mask": torch.tensor(mask) if mask is not None else torch.empty(0),
            "label": torch.tensor(label_int),
            "image_path": image_path,
            "idx": row.get("idx", idx)  # idx 컬럼이 있으면 사용, 없으면 index 반환
        }

class BUSDataLoader(BaseDataLoader):
    """BUS 데이터 로더 (k-fold 지원)"""
    def __init__(self, config, fold_config):
        super().__init__(config)
        self.df = pd.read_csv(config['data_loader']['csv_path'])
        self.train_folds = fold_config['train']
        self.val_fold = fold_config['val']
        self.test_fold = fold_config.get('test') # test는 evaluation.py에서만 사용
        self.setup()

    def setup(self):
        from data_loader.bus_augmentations import get_ultrasound_augmentations
        is_upscale = self.config['data_loader'].get('is_upscale', False)
        
        train_df = self.df[self.df['train_test'].isin(self.train_folds)]
        val_df = self.df[self.df['train_test'] == self.val_fold]
        
        train_transform = get_ultrasound_augmentations() if self.config['data_loader'].get('use_augmentation', True) else None
        
        self.train_dataset = BUSImageDataset(train_df, is_upscale, train_transform)
        self.val_dataset = BUSImageDataset(val_df, is_upscale, None)
        
        if self.test_fold:
            test_df = self.df[self.df['train_test'] == self.test_fold]
            self.test_dataset = BUSImageDataset(test_df, is_upscale, None)

        self.train_loader = DataLoader(self.train_dataset, **self.config['data_loader']['args'])
        self.val_loader = DataLoader(self.val_dataset, **self.config['data_loader']['args'])
        if self.test_fold:
            self.test_loader = DataLoader(self.test_dataset, **self.config['data_loader']['args'])
