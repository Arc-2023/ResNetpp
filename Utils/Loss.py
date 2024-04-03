import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, pred, truth):
        # 如果为double 就报错

        # print(f'Pred: {pred.shape}; Truth: {truth.shape}')

        # BCE loss
        ce_loss = nn.BCELoss()(pred, truth)

        pred = pred.view(-1)

        truth = truth.view(-1)
        # Dice Loss
        num = truth.size(0)
        # 为了防止除0的发生
        smooth = 1
        probs = F.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = truth.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num

        return ce_loss + score


ALPHA = 0.8
GAMMA = 2


class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.cross = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        print(f'Inputs dtype: {inputs.dtype}')
        print(f'Targets dtype: {targets.dtype}')
        targets = torch.squeeze(targets, 1)
        # print(f'Inputs: {inputs.shape}; Targets: {targets.shape}')

        CE = self.cross(inputs, torch.gt(targets, 0.1).long())
        CE_EXP = torch.exp(-CE)
        focal_loss = alpha * (1 - CE_EXP) ** gamma * CE

        # flatten label and prediction tensors
        inputs = inputs[:, 1, :, :].reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        Dice_CE = focal_loss + dice_loss
        # + dice_loss

        return Dice_CE
