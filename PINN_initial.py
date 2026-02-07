import torch
import torch.nn as nn
from torchdiffeq import odeint
import deepxde as dde
import numpy as np
from .utils import system_dynamics
from .initial_model import NeuralODE
from .data import Data

#Aliev Panfilov initial equation

class Geometry():
    def __init___(self, min_x, min_y, max_x, max_y, min_t, max_t):
        self.geom = dde.geometry.Rectangle([min_x,min_y], [max_x,max_y])
        self.timedomain = dde.geometry.TimeDomain(min_t, max_t)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)

class PINN():  #Not declared as nn.Module. Subclasses are there.
    def __init__(self):
        self.dynamics = system_dynamics()
        self.neuralode = NeuralODE()

    def random_pde(self):
        return np.zeros()

    def aliev_panfilov(self,x, y):
        u, r = y[:, 0:1], y[:, 1:2]
        grad_u = dde.grad.jacobian(y, x, i=0, j=0)
        grad2_u_xx = dde.grad.hessian(y, x, i=0, j=0)
        grad2_u_yy = dde.grad.hessian(y, x, i=0, j=1)
        u_t = dde.grad.jacobian(y, x, i=0, j=2)
        diffusion = self.dynamics.D * (grad2_u_xx + grad2_u_yy)
        reaction_u = -self.dynamics.k * u * (u - self.dynamics.a) *(u - 1) -u* r
        eq_u = u_t - diffusion -reaction_u
        r_t = dde.grad.jacobian(y, x, i=1, j=2)
        eq_r = r_t - (self.dynamics.epsilon + self.dynamics.mu_1*r / (self.dynamics.mu_2 + u)) * (self.dynamics.k * u * (self.dynamics.a + 1 - u) - r)
        return [eq_u, eq_r]

    def load_data(self):
        self.geometry = Geometry(self.dynamics.center1, self.dynamics.center2, self.dynamics.r1, self.dynamics.r2, self.dynamics.t1, self.dynamics.t2)
        self.generated_data  = Data(self.aliev_panfilov, self.geometry.geomtime).ddedata

    def forward(self):
        self.neuralode.forward()

    def boundary_loss(self):
        pass

    def initial_loss(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def normal_loss(self, predictions, truth):
        mse_loss = nn.MSELoss()
        return mse_loss(predictions, truth)

    def total_loss(self):
        pass

    def train(self):
        pass
