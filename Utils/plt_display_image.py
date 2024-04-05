import random
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2

transformation = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.ToPILImage(),
])


def display_dynamicly(axs, img, label, pred):
    # image : b 1 512 512
    # lable: b 1 512 512
    random_index = random.randint(0, pred.shape[0] - 1)
    axs[0, 0].imshow(transformation(pred[random_index][0]), cmap='gray')
    axs[0, 0].title.set_text('Prediction: 0')
    axs[0, 1].imshow(transformation(pred[random_index][1]), cmap='gray')
    axs[0, 1].title.set_text('Prediction: 1')
    axs[0, 2].imshow(transformation(pred[random_index][2]), cmap='gray')
    axs[0, 2].title.set_text('Prediction: 2')
    axs[1, 0].imshow(transformation(label[random_index]), cmap='gray')
    axs[1, 0].title.set_text('Label')
    axs[1, 1].imshow(transformation(img[random_index]))
    axs[1, 1].title.set_text('Image')
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
