import torch.nn as nn
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class ExampleNet(torch.nn):
    """
    This Neural Network does nothing! Woohoo!!!!
    """
    def __init__(self):
        super(ExampleNet, self).__init__()

    def forward(self, x):
        return x