import cv2
import numpy as np
from typing import Tuple, Callable, Optional

class BUSPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        self.target_size = target_size
        self.target_width, self.target_height = target_size
        self.transform = transform

    def add_padding(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = max(0, target_height - h)
        pad_w = max(0, target_width - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=0)
        return padded

    def aspect_ratio_preserving_with_padding(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ):
        h, w = image.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = None
        padded_image = self.add_padding(resized_image, self.target_width, self.target_height)
        padded_mask = self.add_padding(resized_mask, self.target_width, self.target_height) if resized_mask is not None else None
        return padded_image, padded_mask

    def process_single_image(
        self, image_path: str, mask_path: Optional[str] = None, label: Optional[str] = None
    ):
        image = cv2.imread(image_path, 0)  # Grayscale
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        mask = None
        if mask_path:
            mask = cv2.imread(mask_path, 0)
        image, mask = self.aspect_ratio_preserving_with_padding(image, mask)
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        return image, mask

    def process_dataset(
        self, df, image_col: str = 'image_path', mask_col: str = 'mask_path'
    ):
        results = []
        for idx, row in df.iterrows():
            image_path = row[image_col]
            mask_path = row.get(mask_col, None)
            label = row.get('label', None)
            try:
                image, mask = self.process_single_image(image_path, mask_path, label)
                results.append({
                    'image': image,
                    'mask': mask,
                    'label': label,
                    'original_path': image_path
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return results