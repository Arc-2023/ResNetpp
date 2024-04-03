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

console = Console()


class TBADataset(Dataset):
    def __init__(self, csv_path: str = base_TBAD_csv_path):
        self.df = pd.read_csv(csv_path, delimiter=',')
        self.transformOfTest = transformation.Compose([
            transformation.PILToTensor(),
            transformation.Resize((384, 384)),
            transformation.Lambda(lambda x: x / 255),
            transformation.ToDtype(torch.float),
        ])
        self.transs = transformation.Compose([
            transformation.ToImage(),
            transformation.ToDtype(torch.float16, scale=True),
            # transformation.Normalize([0], [255.0]),
            transformation.Resize((256, 256)),
            transformation.ConvertImageDtype(torch.float),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_dict = self.df.iloc[idx]

        img = nib.load(data_dict['img'])
        img_data = img.get_fdata()
        mask = nib.load(data_dict['mask'])
        mask_data = mask.get_fdata()
        # console.log(img.header)
        # labels: 0.,1.,2.
        return {'img': img_data, 'mask': mask_data}


if __name__ == '__main__':
    dataset = TBADataset()
    img_dict = dataset[9]
    print(img_dict['img'].shape)
    print(img_dict['mask'].shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img_dict['img'][:, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(img_dict['mask'][:, :, 0])
    plt.show()
    mask = img_dict['mask'][:, :, 0]
    console.log(np.unique(mask))
