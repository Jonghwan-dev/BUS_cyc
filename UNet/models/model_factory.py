# models/model_factory.py

import torch
import torch.nn as nn
from .base_Unet import UNet
from .AttUnet import AttentionUNet

# config의 arch 이름과 실제 모델 클래스를 매핑
MODELS = {
    "UNet": UNet,
    "AttentionUNet": AttentionUNet,
}

def get_model(config, pretrained_weights_path=None):
    """
    config 파일에 따라 모델을 생성하고, 필요시 사전 훈련된 가중치를 로드합니다.
    """
    arch = config['arch']
    model_class = MODELS.get(arch)
    if not model_class:
        raise KeyError(f"Model architecture '{arch}' is not found. Available models: {list(MODELS.keys())}")

    # 1. 기본 모델 구조 생성
    model = model_class(**config['model'])

    # 2. 사전 훈련된 가중치가 있으면 로드 (전이 학습 또는 평가 시 사용)
    if pretrained_weights_path:
        print(f"Loading pretrained weights from: {pretrained_weights_path}")
        # final_conv 레이어의 크기가 다를 수 있으므로 strict=False 사용
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=torch.device('cpu')), strict=False)

    # 3. 전이 학습 시 인코더 동결 (freeze)
    transfer_config = config.get('transfer_learning', {})
    if transfer_config.get('freeze_encoder', False):
        print("Freezing encoder weights.")
        for name, param in model.named_parameters():
            if name.startswith('first_conv') or name.startswith('down'):
                param.requires_grad = False
    
    return model