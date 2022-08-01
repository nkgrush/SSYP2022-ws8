from torch import nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.model(x)
        return x
