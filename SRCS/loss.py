import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        image_loss = self.mse_loss(out_images, target_images)
        return image_loss

if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)

