import os.path

import torch
import torchvision.io
from torchvision.transforms import v2 as transformation
from torch.utils.data import Dataset
import pandas as pd
import Resconfig as Rconfig
from rich.console import Console
from matplotlib import pyplot as plt
from PIL import Image

console = Console()


class TempDataset(Dataset):
    def __init__(self, csv_name: str = r"metadata.csv", split: str = "train"):
        self.df = pd.read_csv(os.path.join(Rconfig.base_path, csv_name), delimiter=',')
        # self.df = self.df.iloc[1:100]
        self.df = self.df[self.df["split"] == split]
        self.transformOfTrain = transformation.Compose([
            # transformation.ToTensor(),
            transformation.PILToTensor(),
            transformation.Resize((384, 384)),
            transformation.Lambda(lambda x: x / 255),
            transformation.ToDtype(torch.float),
        ])
        self.transformOfTest = transformation.Compose([
            transformation.PILToTensor(),
            transformation.Resize((384, 384)),
            transformation.Lambda(lambda x: x / 255),
            transformation.ToDtype(torch.float),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        png_img_path: str = self.df.iloc[idx]['tiff_image_path']
        png_label_path: str = self.df.iloc[idx]['tif_label_path']
        png_img = Image.open(os.path.join(Rconfig.base_path, png_img_path))
        png_label = Image.open(os.path.join(Rconfig.base_path, png_label_path))

        # Convert the images to PyTorch tensors
        png_img_tensor: torch.Tensor = self.transformOfTrain(png_img)
        png_label_tensor: torch.Tensor = self.transformOfTest(png_label)
        return {"image": png_img_tensor, "label": png_label_tensor}

    def test(self, img_dict: dict[str, torch.Tensor]):
        print(img_dict['image'].shape)
        print(img_dict['label'].shape)
        plt.subplot(1, 2)
        plt.imshow(img_dict['image'].permute(1, 2, 0))
        plt.imshow(img_dict['label'].permute(1, 2, 0))
        plt.show()


# print(self.df.tail(5)['split'])


if __name__ == "__main__":
    dataset = TempDataset()
    print(len(dataset))
    dataset.test(dataset[0])
