
import os
import numpy as np
import pandas as pd
from PIL import Image
import splitfolders as sf
import matplotlib.pyplot as plt
from data_read import read_data_Yap, read_data_AlDh, read_data_UC

## Load data

df_yap = read_data_Yap("Yap2018", "./datasets/Yap2018/original", "./datasets/Yap2018/GT", "./datasets/Yap2018/DatasetB.xlsx")
df_aldh = read_data_AlDh("Al-Dhabyani2020", "./datasets/Al-Dhabyani2020")
df_uc = read_data_UC("BUS_UC", "./datasets/BUS_UC")

df_all = pd.concat([df_yap, df_aldh, df_uc], ignore_index=True)
df_all["idx"] = range(len(df_all)) 
df_all = df_all[["idx", "label", "dataset_name", "image_path", "mask_path", "width", "height", "meta"]]
df_all.to_csv("./data/datasets.csv", index=False)
