import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import sys

# 프로젝트 루트 디렉토리 추가
sys.path.append('..')
from data.unaligned_dataset import UnalignedDataset

class Args:
    def __init__(self):
        # 절대 경로 사용
        self.dataroot = os.path.abspath('./data/USenhaned23')
        print(f"데이터셋 경로: {self.dataroot}")
        
        # 경로 존재 확인
        if not os.path.exists(self.dataroot):
            raise ValueError(f"데이터셋 경로가 존재하지 않습니다: {self.dataroot}")
            
        # trainA와 trainB 폴더 확인
        self.dir_A = os.path.join(self.dataroot, 'trainA')
        self.dir_B = os.path.join(self.dataroot, 'trainB')
        
        print(f"Domain A 경로: {self.dir_A}")
        print(f"Domain B 경로: {self.dir_B}")
        
        if not os.path.exists(self.dir_A):
            raise ValueError(f"trainA 폴더가 존재하지 않습니다: {self.dir_A}")
        if not os.path.exists(self.dir_B):
            raise ValueError(f"trainB 폴더가 존재하지 않습니다: {self.dir_B}")
            
        self.phase = 'train'  # 학습 데이터 사용
        self.load_size = 286  # 로드 크기
        self.crop_size = 256  # 크롭 크기
        self.preprocess = 'resize_and_crop'  # 전처리 방식
        self.no_flip = False  # 수평 플립 사용
        self.serial_batches = False  # 랜덤 배치 사용
        self.max_dataset_size = float("inf")  # 최대 데이터셋 크기
        self.direction = 'AtoB'  # 변환 방향
        self.input_nc = 3  # 입력 채널 수
        self.output_nc = 3  # 출력 채널 수
        self.dataset_mode = 'unaligned'  # 데이터셋 모드

def tensor_to_image(tensor):
    """텐서를 이미지로 변환"""
    # [-1, 1] 범위를 [0, 1]로 변환
    img = (tensor + 1) / 2.0
    # 채널 순서 변경 (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0)
    return img.cpu().numpy()

def show_images(real_A, fake_B, real_B, fake_A):
    """이미지 시각화"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(tensor_to_image(real_A))
    plt.title('Real A')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(tensor_to_image(fake_B))
    plt.title('Fake B')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(tensor_to_image(real_B))
    plt.title('Real B')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(tensor_to_image(fake_A))
    plt.title('Fake A')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_preprocessing(image_path, opt):
    """전처리 단계별 이미지 시각화"""
    # 원본 이미지 로드
    img = Image.open(image_path).convert('RGB')
    
    # 전처리 단계별 변환
    resize = transforms.Resize(opt.load_size)
    crop = transforms.RandomCrop(opt.crop_size)
    flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # 각 단계별 이미지 저장
    images = [img]
    titles = ['Original']
    
    # 리사이즈
    img_resized = resize(img)
    images.append(img_resized)
    titles.append('Resized')
    
    # 크롭
    img_cropped = crop(img_resized)
    images.append(img_cropped)
    titles.append('Cropped')
    
    # 플립
    img_flipped = flip(img_cropped)
    images.append(img_flipped)
    titles.append('Flipped')
    
    # 텐서 변환 및 정규화
    img_tensor = normalize(to_tensor(img_flipped))
    images.append(tensor_to_image(img_tensor))
    titles.append('Normalized')
    
    # 시각화
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i+1)
        if isinstance(img, np.ndarray):
            plt.imshow(img)
        else:
            plt.imshow(np.array(img))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # 옵션 설정
    opt = Args()
    
    # 데이터셋 생성
    dataset = UnalignedDataset(opt)
    print(f'데이터셋 크기: {len(dataset)}')
    
    # 데이터 샘플 로드
    data = dataset[0]
    real_A = data['A']
    real_B = data['B']
    
    # 이미지 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_image(real_A))
    plt.title('Domain A')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_image(real_B))
    plt.title('Domain B')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 전처리 파이프라인 시각화
    visualize_preprocessing(data['A_paths'], opt)

if __name__ == '__main__':
    main() 