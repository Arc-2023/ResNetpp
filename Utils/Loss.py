import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, rearrange


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


ALPHA = 0.7
GAMMA = 1
pp = 2
smoo = 1
epsilon = 1e-6


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

    def __init__(self):
        super(SoftDiceFocalLoss, self).__init__()
        self.p = pp
        self.smooth = smoo
        self.cross = nn.CrossEntropyLoss()
        self.focal = focal_loss(alpha=[0.2, 0.8, 0.8])

    def forward(self, inputs, targets):
        # inputs: b 3 h w
        # targets: b 1 h w   (0~3(int) in 1 dim)
        # targets = targets.squeeze(1)
        # print(
        #     f'Inputs: {inputs.shape}; Targets: {targets.shape};targets max: {torch.max(targets)}; targets min: {torch.min(targets)}'))
        # label : b 512 512
        # inputs: b 3 512 512
        targets = torch.squeeze(targets, 1)
        targets = torch.where(targets.eq(3), 0, targets)
        targets = torch.where(targets > 3, 0, targets)
        inputs = rearrange(inputs, 'b c h w -> b h w c')
        # print(f'Targets: {targets.shape}; Inputs: {inputs.shape}')
        CE = self.focal(inputs, targets)
        # # flatten label and prediction tensors
        # inputs = (inputs[:, 1, :, :] + inputs[:, 2, :, :] + inputs[:, 3, :, :]).reshape(-1)

        # targets: torch.Tensor = torch.unsqueeze(targets, 1)
        # Iterate over each unique value
        # b 3 512 512
        # b 3 512 512
        target_dice = torch.stack(
            [torch.where(targets.eq(value), 1, 0) for value in range(3)], dim=1)

        # dice_loss = self.Dice(inputs, target_dice)
        inputs = rearrange(inputs, 'b h w c -> b c h w')
        dice_loss = 0
        for i in range(inputs.shape[1]):  # iterate over each channel
            input_channel = inputs[:, i, :, :]
            target_channel = target_dice[:, i, :, :]
            # print(f'{i} counts in target is {torch.sum(target_channel)}')
            dice_loss = self.Dice(input_channel, target_channel)

        total_loss = dice_loss / 3 + CE

        # + dice_loss

        return total_loss

    def old_focal_loss(self, inputs, targets):
        CE = self.cross(inputs, targets)
        CE_EXP = torch.exp(-CE)
        focal_loss = ALPHA * (1 - CE_EXP) ** GAMMA * CE
        return focal_loss

    def GenerilizeDiceLoss(self, dice_loss, input_channel, target_channel):
        wei = torch.sum(target_channel, axis=[0, 2, 3])  # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(input_channel * target_channel, axis=[0, 2, 3]))
        union = torch.sum(wei * torch.sum(input_channel + target_channel, axis=[0, 2, 3]))
        dice_loss += 1 - (2. * intersection) / (union + epsilon)
        return dice_loss

    def Dice(self, inputs, target_dice):
        # inputs = torch.sigmoid(inputs)
        # target_dice = torch.sigmoid(target_dice)
        # print(f'Probs: {probs.shape}; Target: {target.shape}')
        # b 3 256 256 - b 256 256
        probs = torch.reshape(inputs, (-1,))
        target = torch.reshape(target_dice, (-1,))
        # print(f'Probs: {probs.shape}; Target: {target.shape}')
        numer = (probs * target).sum()
        denor = (probs.pow(self.p) + target.pow(self.p)).sum()
        dice_loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return dice_loss


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.reshape(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.reshape(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.reshape(-1, 1))
        alpha = self.alpha.gather(0, labels.reshape(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
