import albumentations as A

# 초음파 이미지에 적합한 증강만 포함

def get_ultrasound_augmentations():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),  # -10~+10도 회전
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.4),
        A.CoarseDropout(p=0.2),
    ]) 
    return transform