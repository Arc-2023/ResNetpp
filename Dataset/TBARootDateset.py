import os.path
import torch
import torchvision.io
from torchvision.transforms import v2 as transformation
from torch.utils.data import Dataset
import pandas as pd
from Resconfig import base_TBAD_csv_path
from rich.console import Console
from matplotlib import pyplot as plt
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

console = Console()


class TBADaRoottaset(Dataset):
    def __init__(self, csv_path: str = base_TBAD_csv_path):
        self.df: pd.DataFrame = pd.read_csv(csv_path, delimiter=',')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # console.log({"image": row['img'], 'mask': row['mask']})
        img_path = os.path.join(row['img'])
        mask_path = os.path.join(row['mask'])
        return {"image": img_path, 'mask': mask_path}


if __name__ == '__main__':
    dataset = TBADaRoottaset()
    img_dict = dataset[300]
    print(img_dict['image'].shape)
    print(img_dict['label'].shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img_dict['image'].squeeze())
    plt.subplot(1, 2, 2)
    plt.imshow(img_dict['label'].squeeze())
    plt.show()
    mask = img_dict['image']
    console.log(torch.max(mask))

    # test_mat = torch.randn((512, 512, 100), dtype=torch.float16).to('cuda')
    #
    # transs = transformation.Compose([
    #     transformation.ToImage(),
    #     transformation.ToDtype(torch.float16, scale=True),
    #     # transformation.Normalize([0], [255.0]),
    #     transformation.Lambda(lambda x: x / 255),
    #     # transformation.Resize((256, 256)),
    #     transformation.ConvertImageDtype(torch.float),
    # ])
