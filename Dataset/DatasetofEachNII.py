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
from rich.console import Console

console = Console()


class DatasetofEachNII(Dataset):
    def __init__(self, nii_image_path: str, nii_label_path: str):
        console.print(f'[bold green]Loading {nii_image_path} and {nii_label_path}[/bold green]')
        img = nib.load(''.join(nii_image_path))
        self.img_data = img.get_fdata()

        mask = nib.load(''.join(nii_label_path))
        self.mask_data = mask.get_fdata()
        if self.mask_data.ndim < self.img_data.ndim or self.mask_data.shape[2] < self.img_data.shape[2]:
            # mask_data = np.zeros_like(img_data) + mask_data
            self.mask_data = np.pad(self.mask_data,
                                    (0, self.img_data.shape[2] - self.mask_data.shape[2]),
                                    'constant',
                                    constant_values=0)

        self.transformOfTest = transformation.Compose([
            transformation.PILToTensor(),
            transformation.Resize((384, 384)),
            transformation.Lambda(lambda x: x / 255),
            transformation.ToDtype(torch.float),
        ])
        self.trans_train_data = transformation.Compose([
            transformation.ToImage(),
            transformation.ToDtype(torch.float16, scale=True),
            transformation.Normalize([125], [100.0]),
            # transformation.Lambda(lambda x: x / 255),
            transformation.Resize((256, 256)),
            # transformation.Resize((384, 384)),
            transformation.ConvertImageDtype(torch.float),
        ])
        self.trans_label = transformation.Compose([
            transformation.ToImage(),
            # transformation.Resize((384, 384)),
            transformation.Resize((256, 256)),
            transformation.ToDtype(torch.long)
        ])

    def __len__(self):
        return self.img_data.shape[2]

    def __getitem__(self, idx):
        image = torch.as_tensor(self.img_data[:, :, idx])
        mask_d = torch.as_tensor(self.mask_data[:, :, idx])
        return {'image': self.trans_train_data(image), 'label': self.trans_label(mask_d)}
