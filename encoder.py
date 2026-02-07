from torchdiffeq import odeint
import numpy as np
import deepxde as dde
import torch.nn as nn
from .data import Data
import torch

class Encoder():
    def __init__(self, times, data:Data):
        self.times =times
        self.data = data
        self.forward_layers = [
            nn.Conv2d(2, 16, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1)
        ]
        self.backward_layers = [
            nn.ConvTranspose2d(48, 32, 3, 2, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.SiLU(),
            nn.Conv2d(16, 2, 3, 1, 1)
        ]
        self.encoder=nn.Sequential(*self.forward_layers)
        self.decoder = nn.Sequential(*self.backward_layers)
        self.data = Data()

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, x):
        return self.decoder.forward(x)

    def train(self, epochs):
        for epoch in range(0, epochs):
            for t in range(0, len(self.times)):
                self.encode(torch.tensor([self.data.Vsav, self.data.Wsav]))
                self.decode()
