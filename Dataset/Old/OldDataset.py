import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from einops import rearrange
from torchvision.transforms import v2
from rich.console import Console

console = Console()
transs = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float16, scale=True),
    # v2.Normalize([0], [255.0]),
    v2.Resize((256, 256)),
    v2.ConvertImageDtype(torch.float),
])


class OldDataset(Dataset):
    def __init__(self, csv_path):
        super(Dataset, self).__init__()
        self.data = pd.read_csv(csv_path, dtype=str)

    def __len__(self):
        # print(f'len:{len(self.data)}')
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]

        img = nib.load(data_dict['img']).get_fdata()

        mask = nib.load(data_dict['mask']).get_fdata()
        # 192 156 12
        return transs(img), transs(mask), data_dict['img'], data_dict['mask']
