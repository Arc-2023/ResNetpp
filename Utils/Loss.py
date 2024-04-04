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


ALPHA = 0.95
GAMMA = 2
pp = 2
smoo = 1


class SoftDiceFocalLoss(nn.Module):
    """
        SoftDiceFocalLoss类用于计算Soft Dice Focal损失。

        参数:
        weight: 一个可选的权重张量，用于在计算损失时对每个类别进行加权。
        size_average: 一个布尔值，如果为True，则损失将通过平均所有元素来计算；如果为False，则损失将通过对所有元素求和来计算。

        方法:
        forward(inputs, targets, alpha=ALPHA, gamma=GAMMA): 计算Soft Dice Focal损失。
        inputs: 输入张量，形状为[b, 4, h, w]。
        targets: 目标张量，形状为[b, 1, h, w]，其中每个元素的值在0~3之间。
        alpha: Focal损失的alpha参数，默认值为0.8。
        gamma: Focal损失的gamma参数，默认值为2。
        """
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceFocalLoss, self).__init__()
        self.p = pp
        self.smooth = smoo
        self.cross = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        # print(f'Inputs dtype: {inputs.dtype}')
        # print(f'Targets dtype: {targets.dtype}')
        # inputs: b 4 h w
        # targets: b 1 h w   (0~3(int) in 1 dim)
        targets = torch.squeeze(targets, 1)
        print(
            f'Inputs: {inputs.shape}; Targets: {targets.shape};targets max: {torch.max(targets)}; targets min: {torch.min(targets)}')

        CE = self.cross(inputs, targets)
        CE_EXP = torch.exp(-CE)
        focal_loss = alpha * (1 - CE_EXP) ** gamma * CE

        # # flatten label and prediction tensors
        # inputs = (inputs[:, 1, :, :] + inputs[:, 2, :, :] + inputs[:, 3, :, :]).reshape(-1)

        unique_values = torch.unique(targets)
        targets = torch.unsqueeze(targets, 1)
        # Iterate over each unique value
        target_dice = None
        for value in unique_values:
            # print(f'Value: {value}')
            target = torch.where(targets == value, torch.as_tensor(1), torch.as_tensor(0))
            if target_dice is None:
                target_dice = target
            else:
                target_dice = torch.cat((target_dice, target), 1)
        target_dice = torch.nn.functional.pad(target_dice, (0, 0, 0, 0, 0, 4 - target_dice.shape[1]), 'constant', 0)

        probs = torch.sigmoid(inputs)
        target = torch.sigmoid(target_dice)
        # print(f'Probs: {probs.shape}; Target: {target.shape}')
        probs = torch.reshape(probs, (-1,))
        target = torch.reshape(target, (-1,))
        numer = (probs * target).sum()
        denor = (probs.pow(self.p) + target.pow(self.p)).sum()
        dice_loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)

        # Store the result
        # intersection = (inputs * targets).sum()
        # dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        Dice_CE = CE + dice_loss
        # + dice_loss

        return Dice_CE
