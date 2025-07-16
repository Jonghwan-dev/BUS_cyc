# evaluation.py

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from models.model_factory import get_model # 수정
from data_loader.data_loaders import BUSDataLoader
from utils import metrics
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fold_config = {'train': [], 'val': '', 'test': 'test'}
    data_loader = BUSDataLoader(config, fold_config)
    test_loader = data_loader.test_loader
    
    all_folds_preds, all_fold_metrics = [], []
    model_dir = config['trainer']['save_dir']
    dataset_name = Path(config['data_loader']['csv_path']).stem
    true_masks = torch.cat([batch['mask'] for batch in test_loader])
    
    for fold_num in range(1, 6):
        fold = f'fold{fold_num}'
        model_path = os.path.join(model_dir, f"{config['arch']}_{dataset_name}_best_fold_{fold}.pth")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model for {fold} not found. Skipping.")
            continue
            
        print(f"\n--- Evaluating Model from {fold} ---")
        # --- 핵심 수정: get_model로 구조를 먼저 만들고, 저장된 가중치를 로드 ---
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting with {fold} model"):
                images = batch['image'].to(device)
                outputs = torch.sigmoid(model(images))
                fold_preds.append(outputs.cpu())
        
        fold_preds_tensor = torch.cat(fold_preds, dim=0)
        all_folds_preds.append(fold_preds_tensor)

        fold_metric_values = {'Dice': metrics.dice_score(fold_preds_tensor, true_masks),
                              'IoU': metrics.iou_score(fold_preds_tensor, true_masks),
                              'HD95': metrics.hd95_batch(fold_preds_tensor, true_masks)}
        fold_metric_values['fold'] = fold
        all_fold_metrics.append(fold_metric_values)
        print(f"Metrics for {fold}: Dice={fold_metric_values['Dice']:.4f}, IoU={fold_metric_values['IoU']:.4f}, HD95={fold_metric_values['HD95']:.4f}")

    if not all_folds_preds:
        print("No models were found to evaluate.")
        return

    print(f"\n{'='*20} INDIVIDUAL FOLD RESULTS {'='*20}")
    metrics_df = pd.DataFrame(all_fold_metrics).set_index('fold')
    print(metrics_df.round(4))
    print(f"\nAverage Metrics across folds:")
    print(metrics_df.mean().round(4))

    print(f"\n{'='*20} ENSEMBLE EVALUATION {'='*20}")
    ensemble_preds = torch.mean(torch.stack(all_folds_preds), dim=0)
    ensemble_metrics = {'Dice': metrics.dice_score(ensemble_preds, true_masks),
                        'IoU (Jaccard)': metrics.iou_score(ensemble_preds, true_masks),
                        'HD95': metrics.hd95_batch(ensemble_preds, true_masks)}
    for name, value in ensemble_metrics.items():
        print(f"Ensemble {name}: {value:.4f}")

    # ================== 시각화 및 저장 코드 추가 ==================
    vis_dir = os.path.join("results", "vis", "ensemble")
    os.makedirs(vis_dir, exist_ok=True)

    # test_loader에서 이미지, 마스크, idx, image_path 모두 추출
    # (test_loader는 shuffle=False로 가정)
    test_images, test_masks, test_idxs, test_img_paths = [], [], [], []
    for batch in data_loader.test_loader:
        test_images.append(batch["image"])
        test_masks.append(batch["mask"])
        test_idxs.extend(batch["idx"])
        test_img_paths.extend(batch["image_path"])
    test_images = torch.cat(test_images, dim=0)
    test_masks = torch.cat(test_masks, dim=0)

    # ensemble_preds: (N, 1, H, W), test_images: (N, 1, H, W)
    for i in range(ensemble_preds.shape[0]):
        img = test_images[i].squeeze().numpy()
        gt = test_masks[i].squeeze().numpy()
        pred = ensemble_preds[i].squeeze().numpy()
        idx = test_idxs[i]
        img_path = test_img_paths[i]
        # 파일명: idx_원본파일명.png
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(vis_dir, f"{idx}_{base_name}.png")

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(gt, cmap='gray')
        plt.title('GT')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title('Pred (ensemble)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    print(f"\n[Ensemble 예측 시각화 결과 저장 완료: {vis_dir}]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch UNet Evaluation')
    parser.add_argument('-c', '--config', default='test_config.json', type=str, help='config file path')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    evaluate(config)