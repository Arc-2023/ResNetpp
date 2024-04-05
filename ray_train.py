import tempfile
from typing import Callable, Union

import dill
import ray
import torch
from PIL._imaging import draw
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Dataset.DatasetofEachNII import DatasetofEachNII
from Utils.Loss import *
from Dataset.TBARootDateset import *
from Utils.plt_display_image import display_dynamicly
from zoo.res_unet_plus import ResUnetPlusPlus
from zoo.res_unet import ResUnet
from zoo.unet import UNetSmall
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as T

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import cloudpickle
import random

console = Console()
args = {
    'epochs': 10,
    'batch_size': 2,
    'amp': False,
    'shuffle': True,
    'in_channel': 1,
    'out_channel': 3,
    'T_0': 1,
    'T_mult': 2,
    'proportion': 0.9,
    'cos': True,
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '2'
torch.manual_seed(3407)


def train_func(params: dict):
    # model = ResUnetPlusPlus(in_channel=params['in_channel'], out_channel=params['out_channel'])
    model = ResUnetPlusPlus(in_channel=params['in_channel'], out_channel=params['out_channel'])
    # model = ray.train.torch.prepare_model(model)
    # loss_fn = SoftDiceLoss()
    loss_fn = SoftDiceFocalLoss()
    # loss_fn = nn.MSELoss()
    opti = torch.optim.RAdam(model.parameters(), lr=0.003)
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

            train_loader = DataLoader(train_set, batch_size=params['batch_size'],
                                      shuffle=params['shuffle'] if not params['is_parallel'] else False, drop_last=True,
                                      collate_fn=col_fn)
            test_loader = DataLoader(test_set, batch_size=params['batch_size'],
                                     shuffle=params['shuffle'] if not params['is_parallel'] else False, drop_last=True,
                                     collate_fn=col_fn)
            # ray_train_loader = ray.train.torch.prepare_data_loader(train_dataloader)
            # ray_test_loader = ray.train.torch.prepare_data_loader(test_dataloader)
            train_loss = 0
            for niidata in tqdm(train_loader, leave=False):
                img = niidata['images']
                label = niidata['labels']
                if params['amp']:
                    with torch.autocast(device_type=device):
                        opti.zero_grad()
                        model.train()
                        pred = model(img)
                        display_dynamicly(axs, img, label, pred)
                        loss_now = loss_fn(pred, label)
                    train_loss += loss_now.item()
                    scaler.scale(loss_now).backward()
                    scaler.step(opti)
                    if params['cos']:
                        schd_lr.step()
                    scaler.update()
                else:
                    opti.zero_grad()
                    model.train()
                    pred = model(img)
                    display_dynamicly(axs, img, label, pred)
                    loss_now = loss_fn(pred, label)
                    train_loss += loss_now.item()
                    loss_now.backward()
                    opti.step()
                    if params['cos']:
                        schd_lr.step()
                print(f'Epoch: {epoch} Loss: {loss_now * params["batch_size"]}')

            test_loss = 0
            for niidatatest in tqdm(test_loader, leave=False):
                with torch.no_grad():
                    img = niidatatest['images']
                    label = niidatatest['labels']
                    model.eval()
                    pred = model(img)
                    loss = loss_fn(pred, label)
                    test_loss += loss.item()
            train_loss /= (len(train_loader) / params['batch_size'])
            test_loss /= (len(test_loader) / params['batch_size'])
            metrics = {"train_loss": train_loss, 'test_loss': test_loss, "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pt")
                )
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )
            if ray.train.get_context().get_world_rank() == 0:
                print(metrics)
            # del test_dataloader
            # del train_dataloader









def col_fn(images):
    # {"image": png_img, "label": png_label}
    # image: 1*512* 512
    inputs: torch.Tensor = torch.stack([image['image'] for image in images])
    labels: torch.Tensor = torch.stack([image['label'] for image in images])
    return {"images": inputs, "labels": labels}


if __name__ == "__main__":
    ray.shutdown()
    ray.init(include_dashboard=False, dashboard_host='0.0.0.0', ignore_reinit_error=True, dashboard_port=8265)

    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={
        "CPU": 4,
        "GPU": 2,
    }, )
    torch_config = ray.train.torch.TorchConfig(backend='gloo')
    trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=args, scaling_config=scaling_config,
                                           torch_config=torch_config)
    result = trainer.fit()
