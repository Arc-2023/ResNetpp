import torch as torch1
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Dataset.DatasetofEachNII import DatasetofEachNII
from Utils.Loss import *
from Dataset.TBARootDateset import *
from zoo.res_unet import ResUnet
from zoo.res_unet_plus import ResUnetPlusPlus
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as T
from Resconfig import params
from Utils.plt_display_image import display_dynamicly
import random

from zoo.unet import UNet

console = Console()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformation = T.Compose([
    T.ToDtype(torch.float32),
    T.ToPILImage(),
])
# torch.manual_seed(3407)


def train_func():
    model = ResUnetPlusPlus(in_channel=params['in_channel'], out_channel=params['out_channel']).to(device)
    loss_fn = SoftDiceFocalLoss()
    opti = torch.optim.RAdam(model.parameters(), params['lr'])
    schd_lr = None

    if params['cos']:
        schd_lr = CosineAnnealingWarmRestarts(opti, T_0=params['T_0'], T_mult=params['T_mult'])

    fig, axs = plt.subplots(2, 3)
    scaler = torch.cuda.amp.GradScaler(init_scale=8192)
    for epoch in tqdm(range(params['epochs']), leave=False):

        rootset = TBADaRoottaset()
        rootloader = DataLoader(rootset, batch_size=1, shuffle=params['shuffle'], drop_last=True)

        for data in tqdm(rootloader, leave=False):
            nii_set = DatasetofEachNII(nii_image_path=data['image'],
                                       nii_label_path=data['mask'])

            train_set, test_set = torch.utils.data.random_split(nii_set, [int(len(nii_set) * params['proportion']),
                                                                          len(nii_set) - int(
                                                                              len(nii_set) * params['proportion'])])

            train_loader = create_data_loader(train_set, params['is_parallel'], params['batch_size'], params['shuffle'],
                                              col_fn)
            test_loader = create_data_loader(test_set, params['is_parallel'], params['batch_size'], params['shuffle'],
                                             col_fn)

            train_loss = 0
            for niidata in tqdm(train_loader, leave=False):
                img = niidata['images'].to(device)
                label = niidata['labels'].to(device)
                if params['amp']:
                    with torch.autocast(device_type=device):
                        loss_now = model_forward(axs, model, img, label, loss_fn, opti)
                    train_loss += loss_now.item()

                    scaler.scale(loss_now).backward()
                    scaler.unscale_(opti)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opti)

                    if params['cos']:
                        schd_lr.step()

                    scaler.update()
                else:
                    loss_now = model_forward(axs, model, img, label, loss_fn, opti)
                    train_loss += loss_now.item()

                    loss_now.backward()
                    opti.step()

                    if params['cos']:
                        schd_lr.step()
                print(f'Epoch: {epoch} Loss: {loss_now * params["batch_size"]}')

            loss = 0

            for niidatatest in tqdm(test_loader, leave=False):
                with torch1.no_grad():
                    img1 = niidatatest['images'].to(device)
                    label1 = niidatatest['labels'].to(device)
                    model.eval()
                    pred = model(img1)
                    loss = loss_fn(pred, label1)
                    loss += loss.item()
            test_loss = loss

            train_loss /= (len(train_loader) / params['batch_size'])
            test_loss /= (len(test_loader) / params['batch_size'])
            metrics = {"train_loss": train_loss, 'test_loss': test_loss, "epoch": epoch}
            print(metrics)
            # del test_dataloader
            # del train_dataloader


def model_forward(axs, model, img, label, loss_fn, opti):
    opti.zero_grad()
    model.train()
    # img b 1 512 512
    pred = model(img)
    # b 3 512 512
    display_dynamicly(axs, img, label, pred)
    loss_now = loss_fn(pred, label)
    return loss_now


def create_data_loader(dataset, is_parallel, batch_size, shuffle, collate_fn):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_parallel else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if not is_parallel else False,
                        drop_last=True, collate_fn=collate_fn, sampler=sampler if is_parallel else None)
    return loader
    # plt.clf()


def col_fn(images):
    # {"image": png_img, "label": png_label}
    # image: b * 1 * 512 * 512
    inputs: torch.Tensor = torch.stack([image['image'] for image in images])
    labels: torch.Tensor = torch.stack([image['label'] for image in images])
    return {"images": inputs, "labels": labels}


if __name__ == "__main__":
    train_func()
