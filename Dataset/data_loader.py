import torch
from typing import Callable, Union
from TBARootDateset import TBADaRoottaset
from DatasetofEachNII import DatasetofEachNII

from torch.utils.data import DataLoader


def get_root_dataloader(param: dict):
    rootset = TBADaRoottaset()
    rootloader = DataLoader(rootset, batch_size=1, shuffle=True, drop_last=True)
    return rootloader


def get_nii_dataloader(param: dict, fn, nii_image_path: str, nii_label_path: str):
    # console.print(f'[bold green]Loading {nii_image_path} and {nii_label_path}[/bold green]')
    nii_set = DatasetofEachNII(nii_image_path, nii_label_path)
    train_set, test_set = torch.utils.data.random_split(nii_set, [int(len(nii_set) * param['proportion']),
                                                                  len(nii_set) - int(
                                                                      len(nii_set) * param['proportion'])])
    train_loader = DataLoader(train_set, batch_size=param['batch_size'], shuffle=param['shuffle'], drop_last=True,
                              collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=param['batch_size'], shuffle=param['shuffle'], drop_last=True,
                             collate_fn=fn)
    return train_loader, test_loader


def getdataloader(param: dict, fn: Callable, root: bool, nii_image_path: str = None, nii_label_path: str = None) -> \
        Union[DataLoader,
        tuple[DataLoader, DataLoader]]:
    if root:
        return get_root_dataloader(param)
    else:
        return get_nii_dataloader(param, fn, nii_image_path, nii_label_path)
