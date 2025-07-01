# -----------------------
# data_read.py
# -----------------------
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
from PIL import Image


def read_data_Yap(dataset_name: str, image_dir: str, mask_dir: str, excel_path: str) -> pd.DataFrame:
    image_root = Path(image_dir).resolve()
    mask_root = Path(mask_dir).resolve()
    excel_df = pd.read_excel(excel_path)
    excel_df.columns = [col.strip().lower() for col in excel_df.columns]
    excel_df = excel_df.loc[:, ~excel_df.columns.str.contains("^unnamed")]

    data = []
    for img_path in image_root.glob("*.png"):
        filename = img_path.stem
        mask_path = mask_root / f"{filename}.png"
        if not mask_path.exists():
            continue
        match = excel_df[excel_df["image"].astype(str).str.zfill(6) == filename]
        label = match.iloc[0].get("type", "unknown").lower() if not match.empty else "unknown"
        meta = match.iloc[0].to_dict() if not match.empty else {}
        meta.pop("image", None)
        with Image.open(img_path) as im:
            width, height = im.size
        data.append({
            "dataset_name": dataset_name,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "label": label,
            "meta": meta,
            "width": width,
            "height": height
        })

    df = pd.DataFrame(data)
    df["idx"] = range(len(df))
    df = df[["idx", "label", "dataset_name", "image_path", "mask_path", "width", "height", "meta"]]
    return df


def merge_masks(mask_paths: List[Path]) -> Image.Image:
    base = np.array(Image.open(mask_paths[0]).convert("L"))
    for p in mask_paths[1:]:
        m = np.array(Image.open(p).convert("L"))
        base = np.maximum(base, m)
    return Image.fromarray(base)


def read_data_AlDh(dataset_name: str, root_dir: str) -> pd.DataFrame:
    root = Path(root_dir).resolve()
    merged_mask_dir = Path("./datasets/Al-Dhabyani2020/Al-Dhabyani2020_mask_merged")
    merged_mask_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for label_folder in ["benign", "malignant", "normal"]:
        img_dir = root / label_folder
        for img_path in img_dir.glob("*.png"):
            filename = img_path.stem
            mask_paths = sorted(img_dir.glob(f"{filename}_mask*.png"))
            if not mask_paths:
                continue
            merged_mask_path = merged_mask_dir / f"{filename}_merged.png"
            merged_mask_path = merged_mask_path.resolve() 
            if not merged_mask_path.exists():
                merged_mask = merge_masks(mask_paths)
                merged_mask.save(merged_mask_path)
            with Image.open(img_path) as im:
                width, height = im.size
            data.append({
                "dataset_name": dataset_name,
                "image_path": str(img_path),
                "mask_path": str(merged_mask_path),
                "label": label_folder,
                "meta": {"merged": True, "merged_mask_count": len(mask_paths)},
                "width": width,
                "height": height
            })
    df = pd.DataFrame(data)
    df["idx"] = range(len(df))
    df = df[["idx", "label", "dataset_name", "image_path", "mask_path", "width", "height", "meta"]]
    return df


def read_data_UC(dataset_name: str, root_dir: str) -> pd.DataFrame:
    root = Path(root_dir).resolve()
    data = []
    for label_folder in ["Benign", "Malignant"]:
        img_dir = root / label_folder / "images"
        mask_dir = root / label_folder / "masks"
        for img_path in img_dir.glob("*.png"):
            filename = img_path.stem
            mask_path = mask_dir / f"{filename}.png"
            if not mask_path.exists():
                continue
            with Image.open(img_path) as im:
                width, height = im.size

            label = label_folder.lower()

            data.append({
                "dataset_name": dataset_name,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "label": label,
                "meta": {},
                "width": width,
                "height": height
            })
    df = pd.DataFrame(data)
    df["idx"] = range(len(df))
    df = df[["idx", "label", "dataset_name", "image_path", "mask_path", "width", "height", "meta"]]
    return df

