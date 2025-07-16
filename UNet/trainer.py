import torch
import numpy as np
import pandas as pd
import wandb
import os
import cv2
from tqdm import tqdm
from utils import metrics
from pathlib import Path

class Trainer:
    """
    k-fold 교차 검증의 단일 fold에 대한 훈련 및 검증을 총괄하는 클래스.
    - [복원] WandB 로깅 이미지 리사이즈로 로딩 속도 최적화
    - [복원] 전체 히스토리 기반의 동적 랭킹으로 최고 모델 저장
    """
    def __init__(self, model, criterion, optimizer, scheduler, device, config, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.epochs = config['trainer']['epochs']
        self.save_dir = config['trainer']['save_dir']
        self.patience = config['trainer']['early_stop']
        self.val_fold = config['data_loader']['val_fold']
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.wandb_project = config['visualization'].get('wandb_project')

        # 랭크 기반 저장을 위한 상태 변수
        self.metric_history = []
        self.best_epoch = 0

        # WandB 시각화용 고정 이미지 저장 변수
        self.fixed_wandb_images = {}
        self.wandb_images_selected = False

    def train(self):
        if self.wandb_project:
            csv_path = self.config['data_loader']['csv_path']
            dataset_name = Path(csv_path).stem
            run_name = f"{self.config['arch']}_{dataset_name}_fold{self.val_fold}"
            wandb.init(project=self.wandb_project, config=self.config, name=run_name, reinit=True)
        
        epochs_no_improve = 0
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch()
            
            # --- [핵심 수정 1] 랭킹 알고리즘 및 저장 로직 복원 ---
            rank_info = self._update_and_get_best_rank(val_loss, val_metrics, epoch)
            
            # 현재 에폭이 전체 히스토리 중 최고 랭크일 경우 모델 저장
            if rank_info['best_epoch'] == epoch + 1:
                self.best_epoch = epoch + 1
                epochs_no_improve = 0
                
                print(f"    -> New best model found!")
                print(f"       - Current Best Epoch: {self.best_epoch}, Combined Rank: {rank_info['best_score']:.2f}")
                
                model_name = f"{self.config['arch']}_{Path(self.config['data_loader']['csv_path']).stem}_best_fold_{self.val_fold}.pth"
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, model_name))
                print(f"       - Model saved to {model_name}")
            else:
                epochs_no_improve += 1
            
            print(f'[Fold {self.val_fold} | Epoch {epoch+1:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_metrics["dice"]:.4f} | Val HD95: {val_metrics["hd95"]:.4f}')

            if self.wandb_project:
                log_data = {'train/loss': train_loss, 'val/loss': val_loss, **{f'val/{k}': v for k, v in val_metrics.items()},
                            'best_epoch_rank_score': rank_info['best_score']}
                if (epoch + 1) % 5 == 0:
                    self._log_fixed_images_prediction(epoch, log_data)
                else:
                    wandb.log(log_data, step=epoch)

            self.scheduler.step(val_metrics['dice'])

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs with no improvement. Best epoch was {self.best_epoch}.")
                break
        
        if self.wandb_project: wandb.finish()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for batch in progress_bar:
            images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        batch_metrics = {'dice': [], 'iou': [], 'hd95': []}
        with torch.no_grad():
            for batch in self.val_loader:
                if not self.wandb_images_selected: self._select_fixed_images(batch)
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                probs, targets = torch.sigmoid(outputs).cpu(), masks.cpu()
                batch_metrics['dice'].append(metrics.dice_score(probs, targets))
                batch_metrics['iou'].append(metrics.iou_score(probs, targets))
                hd95_val = metrics.hd95_batch(probs, targets)
                if not np.isnan(hd95_val): batch_metrics['hd95'].append(hd95_val)
        
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in batch_metrics.items()}
        if not batch_metrics['hd95']: avg_metrics['hd95'] = 999
        return total_loss / len(self.val_loader), avg_metrics

    def _update_and_get_best_rank(self, val_loss, current_metrics, epoch):
        """전체 히스토리를 다시 계산하여 현재 시점의 최고 에폭 정보를 반환"""
        metrics_to_rank = {'epoch': epoch + 1, 'loss': val_loss, **current_metrics}
        self.metric_history.append(metrics_to_rank)
        
        history_df = pd.DataFrame(self.metric_history)
        
        history_df['loss_rank'] = history_df['loss'].rank(ascending=True, method='dense')
        history_df['dice_rank'] = history_df['dice'].rank(ascending=False, method='dense')
        history_df['hd95_rank'] = history_df['hd95'].rank(ascending=True, method='dense')
        
        history_df['score'] = (history_df['loss_rank'] + history_df['dice_rank'] + history_df['hd95_rank']) / 3
        
        best_epoch_idx = history_df['score'].idxmin()
        best_epoch_info = history_df.loc[best_epoch_idx]
        
        return {"best_epoch": int(best_epoch_info['epoch']), "best_score": best_epoch_info['score']}

    def _select_fixed_images(self, batch):
        if self.wandb_images_selected: return
        class_labels, self.fixed_wandb_images = {0: "benign", 1: "malignant", 2: "normal"}, {}
        images, masks, labels = batch['image'], batch['mask'], batch['label']
        for i in range(images.size(0)):
            label = labels[i].item()
            if label in class_labels and label not in self.fixed_wandb_images:
                self.fixed_wandb_images[label] = {'image': images[i].cpu(), 'mask': masks[i].cpu(), 'label_name': class_labels[label]}
        if len(self.fixed_wandb_images) >= len(class_labels):
            self.wandb_images_selected = True
            print(f"Selected {len([v for v in self.fixed_wandb_images.values() if v])} images for logging.")

    def _log_fixed_images_prediction(self, current_epoch, log_data):
        self.model.eval()
        image_log_dict = {}
        log_img_size = (128, 128) # 로깅용 이미지 크기

        for data in self.fixed_wandb_images.values():
            if data is None: continue
            
            image_tensor = data['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_tensor = torch.sigmoid(self.model(image_tensor)).cpu()

            # --- [핵심 수정 2] 로깅용 리사이즈 로직 복원 ---
            img_np = data['image'].squeeze().numpy()
            gt_mask_np = data['mask'].squeeze().numpy()
            pred_mask_np = (pred_tensor.squeeze().numpy() > 0.5).astype(np.uint8)

            img_log = cv2.resize(img_np, log_img_size, interpolation=cv2.INTER_LINEAR)
            gt_log = cv2.resize(gt_mask_np, log_img_size, interpolation=cv2.INTER_NEAREST)
            pred_log = cv2.resize(pred_mask_np, log_img_size, interpolation=cv2.INTER_NEAREST)
            
            # 3채널로 변환
            img_log_rgb = np.stack([img_log]*3, axis=-1)
            gt_log_rgb = np.stack([gt_log*255]*3, axis=-1)
            pred_log_rgb = np.stack([pred_log*255]*3, axis=-1)
            
            label_name = data['label_name']
            image_log_dict[f"Images/{label_name}/Original"] = wandb.Image(img_log_rgb, caption=f"Epoch {current_epoch+1}")
            image_log_dict[f"Images/{label_name}/Ground_Truth"] = wandb.Image(gt_log_rgb, caption=f"Epoch {current_epoch+1}")
            image_log_dict[f"Images/{label_name}/Prediction"] = wandb.Image(pred_log_rgb, caption=f"Epoch {current_epoch+1}")
        
        if image_log_dict:
            log_data.update(image_log_dict)
        
        wandb.log(log_data, step=current_epoch)