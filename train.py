from typing import Callable

import torch
from PIL._imaging import draw
from einops import rearrange
from torch import nn
from Utils.Loss import *
from Dataset.RoadSegDataset import *
from zoo.res_unet_plus import ResUnetPlusPlus
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as T
import random

console = Console()
input_image_shape = [1500, 1500]
epochs = 10
batch_size = 2
device = 'cuda'

transformation = T.ToPILImage()


def main(args: dict = None):
    model = ResUnetPlusPlus(out_channel=2).to(device)
    # loss_fn = SoftDiceLoss()
    loss_fn = DiceFocalLoss()
    # loss_fn = nn.MSELoss()
    opti = torch.optim.RAdam(model.parameters(), lr=0.001)

    fig, axs = plt.subplots(1, 4)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(epochs), leave=True):
        train_dataloader = getdataloader(col_fn, test=False)
        test_dataloader = getdataloader(col_fn, test=True)
        '''
            console.log(f'Train Dataloader: {len(train_dataloader)};\n Test Dataloader: {len(test_dataloader)}')
            raise Exception("Stop here")
            '''
        train_loss = 0
        for data in tqdm(train_dataloader, leave=False):
            img = data['images'].to(device)
            label = data['labels'].to(device)
            with torch.autocast(device_type=device):
                opti.zero_grad()
                model.train()
                pred = model(img)

                display_dynamicly(axs, img, label, pred)
                loss_now = loss_fn(pred, label)
            train_loss += loss_now
            scaler.scale(loss_now).backward()
            scaler.step(opti)
            scaler.update()

            console.log(f'Epoch: {epoch} Loss: {loss_now}')

        test_loss = 0
        for data in tqdm(test_dataloader, leave=False):
            with torch.no_grad():
                img = data['images'].to(device)
                label = data['labels'].to(device)
                model.eval()
                pred = model(img)
                loss = loss_fn(pred, label)
                test_loss = loss.item()
        train_loss /= len(train_dataloader)
        test_loss /= len(test_dataloader)
        console.log(f'Epoch: {epoch} Train Loss: {train_loss} Test Loss: {test_loss}')


def display_dynamicly(axs, img, label, pred):
    random_index = random.randint(0, len(pred) - 1)
    axs[0].imshow(transformation(pred[random_index][1]))
    axs[0].title.set_text('Prediction: 1')
    axs[1].imshow(transformation(pred[random_index][0]))
    axs[1].title.set_text('Prediction: 0')
    axs[2].imshow(transformation(label[random_index]))
    axs[2].title.set_text('Label')
    axs[3].imshow(transformation(img[random_index]))
    axs[3].title.set_text('Image')
    # Pause for a short period, allowing the plot to update
    plt.pause(0.1)
    # Clear the current axes
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[3].cla()


def getdataloader(fn: Callable, test: bool = False) -> DataLoader:
    dataSet = TempDataset(split='train' if not test else 'test')
    dataloader = DataLoader(dataSet, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=fn)
    return dataloader


def test_data_loader(dataloader: DataLoader):
    for idx, data in enumerate(dataloader):
        print(data['images'].shape)
        raise Exception("Stop here: Test Data Loader")


def col_fn(images: dict):
    # {"image": png_img, "label": png_label}
    inputs: torch.Tensor = torch.stack([image['image'] for image in images])
    labels: torch.Tensor = torch.stack([image['label'] for image in images])
    return {"images": inputs, "labels": labels}


if __name__ == "__main__":
    main()
