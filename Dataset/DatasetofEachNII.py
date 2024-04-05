import os.path
import torch
import torchvision.io
from torchvision.transforms import v2
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
from rich.console import Console

console = Console()


class DatasetofEachNII(Dataset):
    def __init__(self, nii_image_path: str, nii_label_path: str):
        console.print(f'[bold green]Loading {nii_image_path} and {nii_label_path}[/bold green]')
        img = nib.load(''.join(nii_image_path))
        self.img_data = img.get_fdata()


        mask = nib.load(''.join(nii_label_path))
        self.mask_data = mask.get_fdata()
        # img :512 512 300
        # mask:512 512 200
        print(f'img_shape:{self.img_data.shape}')
        print(f'mask_shape:{self.mask_data.shape}')
        print(self.mask_data.shape)
        if self.mask_data.shape[2] < self.img_data.shape[2]:
            # mask_data = np.zeros_like(img_data) + mask_data
            self.mask_data = np.pad(self.mask_data,
                                    (0, self.img_data.shape[2] - self.mask_data.shape[2]),
                                    'constant',
                                    constant_values=0)
        print(f'mask_pad_shape:{self.mask_data.shape}')
        self.trans_train_data = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float16, scale=True),
            v2.Normalize([1], [1.0]),
            # v2.Lambda(lambda x: x / 255),
            # v2.Resize((400, 400)),
            # v2.Resize((384, 384)),
            v2.ConvertImageDtype(torch.float),
        ])
        self.trans_label = v2.Compose([
            v2.ToImage(),
            # v2.Resize((384, 384)),
            # v2.Resize((400, 400)),
            v2.ToDtype(torch.long)
        ])

    def __len__(self):
        return self.img_data.shape[2]

    def __getitem__(self, idx):
        image = torch.as_tensor(self.img_data[:, :, idx])
        mask_d = torch.as_tensor(self.mask_data[:, :, idx])
        return {'image': self.trans_train_data(image), 'label': self.trans_label(mask_d)}
