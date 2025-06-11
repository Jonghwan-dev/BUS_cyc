import os
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import KFold

class KFoldDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options for K-Fold cross validation."""
        parser.add_argument('--k_folds', type=int, default=5, help='K-Fold 분할 수')
        parser.add_argument('--current_fold', type=int, default=0, help='현재 fold 번호')
        parser.add_argument('--val_ratio', type=float, default=0.2, help='train 데이터 중 검증 데이터 비율')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        
        # Set paths
        self.lr_dir = os.path.join(opt.dataroot, 'LR')
        self.hr_dir = os.path.join(opt.dataroot, 'HR')
        
        # Get all image filenames
        self.lr_paths = sorted([f for f in os.listdir(self.lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_paths = sorted([f for f in os.listdir(self.hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create k-fold splits
        kf = KFold(n_splits=opt.k_folds, shuffle=True, random_state=42)
        splits = list(kf.split(self.lr_paths))
        
        # Get current fold indices
        train_val_indices, test_indices = splits[opt.current_fold]
        
        # Split train_val into train and validation
        val_size = int(len(train_val_indices) * opt.val_ratio)
        train_indices = train_val_indices[val_size:]
        val_indices = train_val_indices[:val_size]
        
        # Set indices based on phase
        if opt.phase == 'train':
            self.indices = train_indices
        elif opt.phase == 'val':
            self.indices = val_indices
        else:  # test
            self.indices = test_indices
            
        # Get corresponding paths
        self.lr_paths = [self.lr_paths[i] for i in self.indices]
        self.hr_paths = [self.hr_paths[i] for i in self.indices]
        
        # Set transforms
        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        
    def __len__(self):
        return len(self.lr_paths)
    
    def __getitem__(self, index):
        lr_path = os.path.join(self.lr_dir, self.lr_paths[index])
        hr_path = os.path.join(self.hr_dir, self.hr_paths[index])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Apply transforms
        lr_img = self.transform(lr_img)
        hr_img = self.transform(hr_img)
        
        return {'A': lr_img, 'B': hr_img, 'A_paths': lr_path, 'B_paths': hr_path} 
