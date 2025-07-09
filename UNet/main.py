import pandas as pd
import albumentations as A
from UNet.utils.bus_preprocessor import BUSPreprocessor
from trainer import train_unet

# 데이터프레임 준비 (예시)
df = pd.read_csv('/home/army/workspace/BUS_cyc/UNet/data/datasets.csv')  # image_path, mask_path, label 등 포함

# 증강 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
])

# 전처리 및 증강
preprocessor = BUSPreprocessor(target_size=(256, 256), transform=transform)
processed = preprocessor.process_dataset(df)

# 학습
train_unet(processed)