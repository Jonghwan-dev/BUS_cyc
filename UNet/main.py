# main.py

import torch
import os
import argparse
from models.model_factory import get_model
from data_loader.data_loaders import BUSDataLoader
from trainer import Trainer
from utils.losses import ComboLossHD, WeightedCrossEntropyLoss, DiceBCELoss
from parse_config import ConfigParser

def main(config_parser):
    config = config_parser.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_folds = [f'fold{i}' for i in range(1, 6)]
    
    for val_fold in all_folds:
        print(f"\n{'='*20} TRAINING FOLD: {val_fold} {'='*20}")
        config['data_loader']['val_fold'] = val_fold
        
        train_folds = [f for f in all_folds if f != val_fold]
        fold_config = {'train': train_folds, 'val': val_fold}
        
        data_loader = BUSDataLoader(config, fold_config)
        
        # --- 핵심 수정: get_model 함수로 모든 모델 생성 로직 처리 ---
        # config의 transfer_learning 설정에 따라 자동으로 가중치 로드 및 동결
        pretrained_path = config.get('transfer_learning', {}).get('pretrained_path')
        model = get_model(config, pretrained_weights_path=pretrained_path).to(device)
        
        # 손실 함수 선택 로직
        loss_type = config['loss']['type']
        loss_args = config['loss'].get('args', {})
        if loss_type == 'DiceBCELoss':
            criterion = DiceBCELoss(**loss_args)
        elif loss_type == 'ComboLossHD':
            criterion = ComboLossHD(**loss_args)
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' not supported.")
            
        optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['args'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        trainer = Trainer(model, criterion, optimizer, scheduler, device, config, 
                          data_loader.train_loader, data_loader.val_loader)
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    
    config_parser = ConfigParser(parser)
    main(config_parser)