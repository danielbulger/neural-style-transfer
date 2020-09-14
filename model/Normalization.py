import torch
from torch.nn import Module


class Normalization(Module):

    def __init__(self, device, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std
