from typing import Callable, Union

import torch
from PIL._imaging import draw
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Dataset.DatasetofEachNII import DatasetofEachNII
from Utils.Loss import *
from Dataset.TBARootDateset import *
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
batch_size = 3
device = 'cuda'
amp = False
shuffle = True

T_0 = 1
T_mult = 2

proportion = 0.9
cos = True
transformation = T.Compose([
    T.ToDtype(torch.float32),
    T.ToPILImage(),
])


def main(args: dict = None):
    model = ResUnetPlusPlus(in_channel=1, out_channel=4).to(device)
    # loss_fn = SoftDiceLoss()
    loss_fn = SoftDiceFocalLoss()
    # loss_fn = nn.MSELoss()
    opti = torch.optim.RAdam(model.parameters(), lr=0.02)
    schd_lr = CosineAnnealingWarmRestarts(opti, T_0=T_0, T_mult=T_mult)

    fig, axs = plt.subplots(2, 3)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(epochs), leave=False):
        rootloader = getdataloader(col_fn, root=True)
        for data in tqdm(rootloader, leave=False):
            train_dataloader, test_dataloader = getdataloader(col_fn, root=False, nii_image_path=data['image'],
                                                              nii_label_path=data['mask'])
            train_loss = 0
            for niidata in tqdm(train_dataloader, leave=False):
                img = niidata['images'].to(device)
                label = niidata['labels'].to(device)
                if amp:
                    with torch.autocast(device_type=device):
                        opti.zero_grad()
                        model.train()
                        pred = model(img)
                        display_dynamicly(axs, img, label, pred)
                        loss_now = loss_fn(pred, label)
                    train_loss += loss_now
                    scaler.scale(loss_now).backward()
                    scaler.step(opti)
                    if schd_lr:
                        schd_lr.step()
                    scaler.update()
                else:
                    opti.zero_grad()
                    model.train()
                    pred = model(img)
                    display_dynamicly(axs, img, label, pred)
                    loss_now = loss_fn(pred, label)
                    train_loss += loss_now
                    loss_now.backward()
                    opti.step()
                    if schd_lr:
                        schd_lr.step()
                console.log(f'Epoch: {epoch} Loss: {loss_now * batch_size}')

            test_loss = 0
            for niidata in tqdm(test_dataloader, leave=False):
                with torch.no_grad():
                    img = niidata['images'].to(device)
                    label = niidata['labels'].to(device)
                    model.eval()
                    pred = model(img)
                    loss = loss_fn(pred, label)
                    test_loss += loss.item()
            train_loss /= (len(train_dataloader) / batch_size)
            test_loss /= (len(test_dataloader) / batch_size)
            console.log(f'Epoch: {epoch} Train Loss: {train_loss} Test Loss: {test_loss}')
            del test_dataloader
            del train_dataloader


def display_dynamicly(axs, img, label, pred):
    random_index = random.randint(0, pred.shape[0] - 1)
    axs[0, 0].imshow(transformation(pred[random_index][0]))
    axs[0, 0].title.set_text('Prediction: 0')
    axs[0, 1].imshow(transformation(pred[random_index][1]))
    axs[0, 1].title.set_text('Prediction: 1')
    axs[0, 2].imshow(transformation(pred[random_index][2]))
    axs[0, 2].title.set_text('Prediction: 2')
    axs[1, 0].imshow(transformation(label[random_index]))
    axs[1, 0].title.set_text('Label')
    axs[1, 1].imshow(transformation(img[random_index]))
    axs[1, 1].title.set_text('Image')
    axs[1, 2].imshow(transformation(pred[random_index][3]))
    axs[1, 2].title.set_text('Prediction: 3')
    # plt.colorbar()
    # Pause for a short period, allowing the plot to update
    plt.pause(0.1)
    # Clear the current axes
    axs[0, 0].cla()
    axs[0, 1].cla()
    axs[0, 2].cla()
    axs[1, 0].cla()
    axs[1, 1].cla()
    axs[1, 2].cla()
    # plt.clf()


def get_root_dataloader():
    rootset = TBADaRoottaset()
    rootloader = DataLoader(rootset, batch_size=1, shuffle=shuffle, drop_last=True)
    return rootloader


def get_nii_dataloader(fn, nii_image_path: str, nii_label_path: str):
    console.print(f'[bold green]Loading {nii_image_path} and {nii_label_path}[/bold green]')
    nii_set = DatasetofEachNII(nii_image_path, nii_label_path)
    train_set, test_set = torch.utils.data.random_split(nii_set, [int(len(nii_set) * proportion),
                                                                  len(nii_set) - int(len(nii_set) * proportion)])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=fn)
    return train_loader, test_loader


def getdataloader(fn: Callable, root: bool, nii_image_path: str = None, nii_label_path: str = None) -> Union[DataLoader,
tuple[DataLoader, DataLoader]]:
    if root:
        return get_root_dataloader()
    else:
        return get_nii_dataloader(fn, nii_image_path, nii_label_path)


def test_data_loader(dataloader: DataLoader):
    for idx, data in enumerate(dataloader):
        print(data['images'].shape)
        raise Exception("Stop here: Test Data Loader")


def col_fn(images: dict):
    # {"image": png_img, "label": png_label}
    # image: 1*512* 512
    inputs: torch.Tensor = torch.stack([image['image'] for image in images])
    labels: torch.Tensor = torch.stack([image['label'] for image in images])
    return {"images": inputs, "labels": labels}


if __name__ == "__main__":
    main()
